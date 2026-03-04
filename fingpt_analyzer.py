#!/usr/bin/env python
"""FinGPT舆情分析模块 - 加密货币新闻、社交媒体情绪分析和黑天鹅事件检测"""

import os
import sys
import logging

# 配置日志 - 兼容 PyInstaller 打包环境（无终端模式）
log_file = None
try:
    # 如果是打包后的无终端模式，创建日志文件
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.path.dirname(sys.executable), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'fingpt_analyzer.log')
    
    # 配置日志
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a',
            encoding='utf-8'
        )
    else:
        # 开发模式，输出到控制台
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
except Exception:
    # 备用方案：不配置日志，避免崩溃
    pass

# 禁用 transformers 的详细日志
try:
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.ERROR)
except Exception:
    pass

# 设置HuggingFace环境变量
os.environ["HF_HUB_DISABLE_XET"] = "1"

# 尝试加载.env文件以获取HF_TOKEN
try:
    from dotenv import load_dotenv
    
    # 兼容 PyInstaller 打包环境
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)
    
    # 加载.env文件
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    # 如果.env文件中设置了HF_TOKEN，则设置环境变量
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.strip() and not hf_token.startswith("your_"):
        os.environ["HF_TOKEN"] = hf_token
except ImportError:
    pass
except Exception:
    pass

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("警告: transformers库未安装，部分功能受限")

class FinGPTSentimentAnalyzer:
    """FinGPT舆情分析器 - 加密货币新闻和社交媒体情绪分析"""
    
    def __init__(self, 
                 use_local_model: bool = True,
                 model_name: str = "FinGPT/fingpt-forecaster"):
        """
        初始化FinGPT分析器
        
        Args:
            use_local_model: 是否使用本地模型
            model_name: 本地模型名称
        """
        self.use_local_model = use_local_model
        self.model_name = model_name
        
        # 兼容 PyInstaller 打包环境
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(__file__)
        
        # 新闻爬虫缓存文件
        self.news_cache_file = os.path.join(base_dir, "btc_news_cache.json")
        self.sentiment_cache_file = os.path.join(base_dir, "news_sentiment_cache.json")
        
        # 情绪分析模型
        self.sentiment_model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        
        # 事件检测关键词
        self.black_swan_keywords = {
            "regulatory": ["regulation", "ban", "restrict", "crackdown", "lawsuit", "sec", "cfdc"],
            "security": ["hack", "exploit", "breach", "attack", "vulnerability", "rug pull"],
            "economic": ["inflation", "recession", "crisis", "default", "bankrupt", "bailout"],
            "technical": ["bug", "glitch", "downtime", "maintenance", "hard fork", "consensus"],
            "market": ["crash", "flash crash", "manipulation", "wash trading", "insider"]
        }
        
        # 初始化模型
        self._initialize_models()
        
        # 缓存最近的分析结果
        self.cache = {}
        self.cache_ttl = 300  # 5分钟缓存
        
        print(f"FinGPT舆情分析器初始化完成 (本地模型: {use_local_model})")
    
    def _initialize_models(self):
        """初始化情绪分析模型"""
        if self.use_local_model and TRANSFORMERS_AVAILABLE:
            try:
                print("正在加载本地FinGPT情绪分析模型...")
                # 使用较小的模型以节省资源
                # 使用项目目录下的模型
                model_path = os.path.join(base_dir, "models", "fingpt-sentiment")
                model_name = model_path
                try:
                    # 首先尝试仅本地加载
                    self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                    self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
                    print("  ✓ 从本地缓存加载模型成功")
                except Exception as local_e:
                    print(f"  ⚠️ 本地缓存加载失败: {local_e}")
                    print("  ⚠️ 模型未在本地缓存，FinGPT情绪分析功能将不可用")
                    self.use_local_model = False
                    raise
                
                # 移到GPU（如果可用）
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                if self.device == "cuda":
                    self.sentiment_model = self.sentiment_model.to(self.device)
                    print(f"  ✓ 模型移至GPU ({torch.cuda.get_device_name(0)})")
                else:
                    print("  使用CPU运行")
                
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", 
                    model=self.sentiment_model, 
                    tokenizer=self.tokenizer,
                    device=0 if self.device == "cuda" else -1
                )
                print("✓ 本地情绪分析模型加载完成")
            except Exception as e:
                print(f"本地模型加载失败: {e}")
                print("将使用基于规则的情绪分析")
                self.use_local_model = False
        else:
            print("使用基于规则的情绪分析")
    
    def _load_news_from_crawler_cache(self) -> Optional[List[Dict]]:
        """从新闻爬虫缓存加载新闻数据"""
        if not os.path.exists(self.news_cache_file):
            return None
        
        try:
            with open(self.news_cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
                news_list = cache_data.get('news', [])
                print(f"  ✓ 从新闻爬虫缓存加载了{len(news_list)}条新闻")
                return news_list
        except Exception as e:
            print(f"  ⚠️ 加载新闻爬虫缓存失败: {e}")
            return None
    
    def _load_sentiment_cache(self) -> Dict:
        """从情绪分析缓存加载数据"""
        if not os.path.exists(self.sentiment_cache_file):
            return {}
        
        try:
            with open(self.sentiment_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"  ⚠️ 加载情绪分析缓存失败: {e}")
            return {}
    
    def fetch_crypto_news(self, symbol: str = "BTC", limit: int = 20) -> List[Dict]:
        """获取加密货币新闻数据（使用新闻爬虫缓存）
        
        Args:
            symbol: 加密货币符号 (BTC, ETH等)
            limit: 最大新闻数量
            
        Returns:
            新闻数据列表
        """
        news_data = []
        
        # 从新闻爬虫缓存加载
        cached_news = self._load_news_from_crawler_cache()
        if cached_news:
            news_data = cached_news[:limit]
            print(f"  ✓ 使用新闻爬虫缓存的{len(news_data)}条新闻")
        else:
            print("  ⚠️ 新闻爬虫缓存不可用，使用备用数据")
            sample_news = [
                {
                    "title": f"{symbol} Price Update",
                    "content": f"Market news for {symbol}",
                    "source": "NewsCrawler",
                    "published_at": datetime.now().isoformat(),
                    "sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "url": ""
                }
            ]
            news_data = sample_news
        
        return news_data
    
    def fetch_social_sentiment(self, symbol: str = "BTC") -> Dict:
        """获取社交媒体情绪数据（简化版）
        
        Args:
            symbol: 加密货币符号
            
        Returns:
            社交媒体情绪分析结果
        """
        # 简化版本，不再调用外部API
        social_data = {
            "twitter": {
                "sentiment_score": 0.5,
                "mention_count": 1000,
                "trending_topics": [f"#{symbol}", "#crypto"],
                "top_influencers": []
            },
            "reddit": {
                "sentiment_score": 0.5,
                "mention_count": 500,
                "top_subreddits": ["r/CryptoCurrency", f"r/{symbol}"],
                "hot_posts": []
            }
        }
        
        return social_data
    
    def analyze_sentiment(self, text: str) -> Dict:
        """分析文本情绪
        
        Args:
            text: 待分析文本
            
        Returns:
            情绪分析结果
        """
        if not text or len(text.strip()) < 10:
            return {"sentiment": "neutral", "score": 0.0, "confidence": 0.0}
        
        # 使用本地模型分析
        if self.use_local_model and self.sentiment_pipeline:
            try:
                result = self.sentiment_pipeline(text[:512])  # 限制文本长度
                sentiment = result[0]['label'].lower()
                score = result[0]['score']
                
                # 转换为统一的情绪分数 (-1到1)
                if sentiment in ["positive", "bullish"]:
                    sentiment_score = score
                elif sentiment in ["negative", "bearish"]:
                    sentiment_score = -score
                else:
                    sentiment_score = 0.0
                
                return {
                    "sentiment": sentiment,
                    "score": sentiment_score,
                    "confidence": score,
                    "method": "FinGPT模型"
                }
            except Exception as e:
                print(f"模型情绪分析失败: {e}")
                # 回退到基于规则的分析
        
        # 基于规则的情绪分析（备用）
        return self._rule_based_sentiment(text)
    
    def _rule_based_sentiment(self, text: str) -> Dict:
        """基于规则的情绪分析（备用方法）"""
        text_lower = text.lower()
        
        # 积极关键词
        positive_keywords = [
            "bullish", "surge", "rally", "breakout", "gain", "increase", 
            "positive", "optimistic", "strong", "support", "buy", "accumulate",
            "uptrend", "recovery", "rebound", "momentum"
        ]
        
        # 消极关键词
        negative_keywords = [
            "bearish", "drop", "crash", "decline", "loss", "decrease",
            "negative", "pessimistic", "weak", "resistance", "sell", "dump",
            "downtrend", "correction", "plunge", "volatility"
        ]
        
        # 计数关键词
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        # 计算情绪分数
        total = pos_count + neg_count
        if total > 0:
            sentiment_score = (pos_count - neg_count) / total
        else:
            sentiment_score = 0.0
        
        # 确定情绪标签
        if sentiment_score > 0.3:
            sentiment = "positive"
        elif sentiment_score < -0.3:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": sentiment_score,
            "confidence": min(abs(sentiment_score) * 2, 1.0),  # 置信度估计
            "method": "基于规则"
        }
    
    def detect_black_swan_events(self, news_data: List[Dict]) -> Dict:
        """检测黑天鹅事件
        
        Args:
            news_data: 新闻数据列表
            
        Returns:
            黑天鹅事件检测结果
        """
        events = []
        risk_level = "LOW"
        event_types = set()
        
        for news in news_data:
            content = f"{news.get('title', '')} {news.get('content', '')}".lower()
            
            # 检查各类关键词
            for event_type, keywords in self.black_swan_keywords.items():
                for keyword in keywords:
                    if keyword in content:
                        events.append({
                            "type": event_type,
                            "keyword": keyword,
                            "title": news.get("title", ""),
                            "source": news.get("source", ""),
                            "time": news.get("published_at", "")
                        })
                        event_types.add(event_type)
                        break  # 每个新闻只检测一种事件类型
        
        # 确定风险等级
        if len(events) >= 5:
            risk_level = "HIGH"
        elif len(events) >= 2:
            risk_level = "MEDIUM"
        
        # 如果有监管或安全事件，提高风险等级
        if "regulatory" in event_types or "security" in event_types:
            risk_level = "HIGH"
        
        return {
            "events_detected": events,
            "risk_level": risk_level,
            "event_types": list(event_types),
            "event_count": len(events),
            "recommendation": self._get_risk_recommendation(risk_level)
        }
    
    def _get_risk_recommendation(self, risk_level: str) -> str:
        """根据风险等级获取交易建议"""
        recommendations = {
            "LOW": "正常交易，信号有效",
            "MEDIUM": "谨慎交易，减少仓位，增加止损",
            "HIGH": "暂停交易，规避风险，等待市场稳定"
        }
        return recommendations.get(risk_level, "正常交易")
    
    def analyze_market_sentiment(self, symbol: str = "BTC") -> Dict:
        """综合分析市场情绪
        
        Args:
            symbol: 加密货币符号
            
        Returns:
            综合情绪分析结果
        """
        print(f"\n[FinGPT] 开始分析 {symbol} 市场情绪...")
        
        # 检查缓存
        cache_key = f"sentiment_{symbol}_{datetime.now().strftime('%Y%m%d%H')}"
        if cache_key in self.cache:
            cache_time, cached_data = self.cache[cache_key]
            if time.time() - cache_time < self.cache_ttl:
                print(f"  ✓ 使用缓存的情绪分析结果")
                return cached_data
        
        # 获取新闻数据
        print(f"  [1/4] 正在获取新闻数据...")
        news_data = self.fetch_crypto_news(symbol)
        print(f"      ✓ 获取到 {len(news_data)} 条新闻")
        
        # 分析每条新闻的情绪（如果新闻已包含情绪分析则直接使用）
        print(f"  [2/4] 正在分析新闻情绪...")
        news_sentiments = []
        for news in news_data:
            if "sentiment_score" in news:
                # 使用新闻爬虫已分析好的情绪
                sentiment_score = news["sentiment_score"]
                sentiment = news.get("sentiment", "neutral")
                news_sentiments.append(sentiment_score)
            else:
                # 重新分析
                sentiment_result = self.analyze_sentiment(
                    f"{news.get('title', '')} {news.get('content', '')}"
                )
                news["sentiment_analysis"] = sentiment_result
                news_sentiments.append(sentiment_result["score"])
        
        # 计算新闻平均情绪
        if news_sentiments:
            avg_news_sentiment = np.mean(news_sentiments)
            print(f"      ✓ 新闻平均情绪: {avg_news_sentiment:.3f}")
        else:
            avg_news_sentiment = 0.0
            print(f"      ⚠ 没有新闻数据可用")
        
        # 获取社交媒体情绪
        print(f"  [3/4] 正在获取社交媒体数据...")
        social_data = self.fetch_social_sentiment(symbol)
        print(f"      ✓ 社交媒体数据获取完成")
        
        # 结合社交媒体情绪（加权平均）
        twitter_weight = 0.6
        reddit_weight = 0.4
        
        twitter_score = social_data.get("twitter", {}).get("sentiment_score", 0.0)
        reddit_score = social_data.get("reddit", {}).get("sentiment_score", 0.0)
        
        social_sentiment = (twitter_score * twitter_weight + reddit_score * reddit_weight)
        print(f"      社交媒体情绪 - Twitter: {twitter_score:.3f}, Reddit: {reddit_score:.3f}")
        
        # 综合情绪分数（新闻权重0.7，社交媒体权重0.3）
        overall_sentiment = (avg_news_sentiment * 0.7 + social_sentiment * 0.3)
        print(f"      ✓ 综合情绪分数: {overall_sentiment:.3f}")
        
        # 检测黑天鹅事件
        print(f"  [4/4] 正在检测黑天鹅事件...")
        black_swan_analysis = self.detect_black_swan_events(news_data)
        print(f"      ✓ 风险等级: {black_swan_analysis['risk_level']}")
        
        # 确定市场情绪状态
        if overall_sentiment > 0.3:
            market_sentiment = "BULLISH"
        elif overall_sentiment < -0.3:
            market_sentiment = "BEARISH"
        else:
            market_sentiment = "NEUTRAL"
        
        print(f"\n[FinGPT] {symbol} 市场情绪分析完成:")
        print(f"  整体情绪: {market_sentiment}")
        print(f"  情绪分数: {overall_sentiment:.3f}")
        print(f"  风险等级: {black_swan_analysis['risk_level']}")
        print(f"  建议: {black_swan_analysis['recommendation']}")
        
        # 生成最终结果
        result = {
            "symbol": symbol,
            "overall_sentiment": market_sentiment,
            "sentiment_score": float(overall_sentiment),  # -1到1
            "news_sentiment": float(avg_news_sentiment),
            "social_sentiment": float(social_sentiment),
            "news_count": len(news_data),
            "black_swan_analysis": black_swan_analysis,
            "risk_level": black_swan_analysis["risk_level"],
            "recommendation": black_swan_analysis["recommendation"],
            "timestamp": datetime.now().isoformat(),
            "news_samples": news_data[:3],  # 返回前3条新闻作为样本
            "social_data_summary": {
                "twitter_mentions": social_data.get("twitter", {}).get("mention_count", 0),
                "reddit_mentions": social_data.get("reddit", {}).get("mention_count", 0)
            }
        }
        
        # 更新缓存
        self.cache[cache_key] = (time.time(), result)
        
        return result
    
    def filter_trading_signal(self, 
                             trading_signal: Dict, 
                             sentiment_analysis: Dict) -> Dict:
        """使用舆情分析过滤交易信号
        
        Args:
            trading_signal: 原始交易信号（来自Kronos）
            sentiment_analysis: 舆情分析结果
            
        Returns:
            过滤后的交易信号
        """
        if not trading_signal or not sentiment_analysis:
            return trading_signal
        
        # 复制原始信号
        filtered_signal = trading_signal.copy()
        
        # 提取关键信息
        risk_level = sentiment_analysis.get("risk_level", "LOW")
        sentiment_score = sentiment_analysis.get("sentiment_score", 0.0)
        overall_sentiment = sentiment_analysis.get("overall_sentiment", "NEUTRAL")
        recommendation = sentiment_analysis.get("recommendation", "正常交易")
        
        # 信号方向
        signal_direction = trading_signal.get("trend_direction", "NEUTRAL")
        
        # 黑天鹅事件过滤
        if risk_level == "HIGH":
            filtered_signal["signal_valid"] = False
            filtered_signal["filter_reason"] = "黑天鹅事件风险高"
            filtered_signal["filtered_by"] = "FinGPT舆情分析"
            filtered_signal["original_signal_valid"] = trading_signal.get("signal_valid", False)
            return filtered_signal
        
        # 情绪一致性检查
        sentiment_consistent = True
        # 获取信号强度（归一化值）
        signal_strength = trading_signal.get('trend_strength', 0)
        
        if signal_direction == "LONG" and overall_sentiment == "BEARISH":
            sentiment_consistent = False
            # 根据信号强度动态调整权重：高强度信号惩罚较小
            if signal_strength > 0.7:  # 高强度
                sentiment_weight = 0.8  # 轻微惩罚
            elif signal_strength > 0.5:  # 中强度
                sentiment_weight = 0.6  # 中等惩罚
            else:
                sentiment_weight = 0.5  # 标准惩罚
        elif signal_direction == "SHORT" and overall_sentiment == "BULLISH":
            sentiment_consistent = False
            # 根据信号强度动态调整权重：高强度信号惩罚较小
            if signal_strength > 0.7:  # 高强度
                sentiment_weight = 0.8  # 轻微惩罚
            elif signal_strength > 0.5:  # 中强度
                sentiment_weight = 0.6  # 中等惩罚
            else:
                sentiment_weight = 0.5  # 标准惩罚
        else:
            # 情绪一致，增强信号
            if overall_sentiment in ["BULLISH", "BEARISH"]:
                sentiment_weight = 1.2  # 增强20%
            else:
                sentiment_weight = 1.0
        
        # 调整信号强度
        if "trend_strength" in filtered_signal:
            original_strength = filtered_signal["trend_strength"]
            filtered_signal["trend_strength"] = original_strength * sentiment_weight
            filtered_signal["sentiment_adjusted"] = True
        
        # 风险等级调整
        if risk_level == "MEDIUM":
            # 中等风险：降低仓位建议
            filtered_signal["position_size_multiplier"] = 0.5
            filtered_signal["risk_adjusted"] = True
        
        # 记录过滤信息
        filtered_signal["sentiment_analysis"] = {
            "risk_level": risk_level,
            "sentiment_score": sentiment_score,
            "overall_sentiment": overall_sentiment,
            "recommendation": recommendation,
            "sentiment_consistent": sentiment_consistent,
            "sentiment_weight": sentiment_weight
        }
        
        # 更新信号有效性
        if not sentiment_consistent and risk_level == "MEDIUM":
            filtered_signal["signal_valid"] = False
            filtered_signal["filter_reason"] = "情绪不一致且风险中等"
        elif not sentiment_consistent:
            # 仅情绪不一致但风险低，仍可交易但标记警告
            filtered_signal["signal_warning"] = "情绪不一致"
        
        filtered_signal["filtered_by"] = "FinGPT舆情分析"
        filtered_signal["filter_timestamp"] = datetime.now().isoformat()
        
        return filtered_signal
    
    def get_detailed_report(self, symbol: str = "BTC") -> Dict:
        """生成详细的舆情分析报告
        
        Args:
            symbol: 加密货币符号
            
        Returns:
            详细分析报告
        """
        sentiment_analysis = self.analyze_market_sentiment(symbol)
        
        report = {
            "summary": {
                "symbol": symbol,
                "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_sentiment": sentiment_analysis["overall_sentiment"],
                "sentiment_score": sentiment_analysis["sentiment_score"],
                "risk_level": sentiment_analysis["risk_level"],
                "trading_recommendation": sentiment_analysis["recommendation"]
            },
            "news_analysis": {
                "total_news": sentiment_analysis["news_count"],
                "avg_sentiment": sentiment_analysis["news_sentiment"],
                "sample_news": sentiment_analysis["news_samples"]
            },
            "social_analysis": {
                "twitter_sentiment": sentiment_analysis["social_data_summary"].get("twitter_sentiment", 0.0),
                "reddit_sentiment": sentiment_analysis["social_data_summary"].get("reddit_sentiment", 0.0),
                "twitter_mentions": sentiment_analysis["social_data_summary"].get("twitter_mentions", 0),
                "reddit_mentions": sentiment_analysis["social_data_summary"].get("reddit_mentions", 0)
            },
            "risk_analysis": sentiment_analysis["black_swan_analysis"],
            "market_context": {
                "volatility_estimate": "待集成",  # 可集成波动率数据
                "market_trend": "待集成",  # 可集成市场趋势数据
                "correlation_assets": ["待集成"]  # 可集成相关性分析
            }
        }
        
        return report


# 使用示例
if __name__ == "__main__":
    # 创建分析器实例
    analyzer = FinGPTSentimentAnalyzer(
        use_local_model=True  # 使用本地模型（如果可用）
    )
    
    # 分析市场情绪
    print("正在分析BTC市场情绪...")
    sentiment_result = analyzer.analyze_market_sentiment("BTC")
    
    print(f"\n=== 市场情绪分析结果 ===")
    print(f"综合情绪: {sentiment_result['overall_sentiment']}")
    print(f"情绪分数: {sentiment_result['sentiment_score']:.3f}")
    print(f"风险等级: {sentiment_result['risk_level']}")
    print(f"交易建议: {sentiment_result['recommendation']}")
    
    # 黑天鹅事件检测
    black_swan = sentiment_result['black_swan_analysis']
    if black_swan['event_count'] > 0:
        print(f"\n⚠️  检测到{black_swan['event_count']}个潜在风险事件:")
        for event in black_swan['events_detected'][:3]:  # 显示前3个
            print(f"  - {event['type']}: {event['title'][:50]}...")
    
    # 生成详细报告
    report = analyzer.get_detailed_report("BTC")
    print(f"\n📊 详细报告已生成，包含{report['news_analysis']['total_news']}条新闻分析")
    
    # 模拟交易信号过滤
    sample_signal = {
        "trend_direction": "LONG",
        "trend_strength": 0.8,
        "signal_valid": True,
        "price_change_pct": 0.05
    }
    
    filtered_signal = analyzer.filter_trading_signal(sample_signal, sentiment_result)
    print(f"\n=== 信号过滤示例 ===")
    print(f"原始信号: {sample_signal['trend_direction']}, 强度: {sample_signal['trend_strength']}")
    print(f"过滤后信号有效: {filtered_signal.get('signal_valid', '未知')}")
    if 'filter_reason' in filtered_signal:
        print(f"过滤原因: {filtered_signal['filter_reason']}")
