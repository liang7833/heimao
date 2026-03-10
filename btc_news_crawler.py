#!/usr/bin/env python
"""
BTC新闻爬虫模块 - 轻量级多源新闻获取
整合多个加密货币新闻源，提供高质量的BTC相关新闻
集成 FinGPT 情绪分析和 Qwen 翻译功能
"""

import os
import sys
import json
import time
import feedparser
import requests
from datetime import datetime
from typing import List, Dict, Optional
from dataclasses import dataclass
from bs4 import BeautifulSoup
import numpy as np


@dataclass
class NewsItem:
    """新闻数据结构"""
    title: str
    content: str
    source: str
    url: str
    published_at: str
    sentiment: str = "neutral"
    sentiment_score: float = 0.0
    image_url: Optional[str] = None
    title_cn: Optional[str] = None
    content_cn: Optional[str] = None


class BTCNewsCrawler:
    """BTC新闻爬虫 - 多源获取高质量新闻"""
    
    def __init__(self, cache_file: str = None, sentiment_cache_file: str = None):
        """
        初始化新闻爬虫
        
        Args:
            cache_file: 新闻缓存文件路径
            sentiment_cache_file: 情绪分析缓存文件路径
        """
        # 兼容 PyInstaller 打包环境
        if getattr(sys, 'frozen', False):
            base_dir = os.path.dirname(sys.executable)
        else:
            base_dir = os.path.dirname(__file__)
        
        if cache_file is None:
            self.cache_file = os.path.join(base_dir, "btc_news_cache.json")
        else:
            if os.path.isabs(cache_file):
                self.cache_file = cache_file
            else:
                self.cache_file = os.path.join(base_dir, cache_file)
        
        if sentiment_cache_file is None:
            self.sentiment_cache_file = os.path.join(base_dir, "news_sentiment_cache.json")
        else:
            if os.path.isabs(sentiment_cache_file):
                self.sentiment_cache_file = sentiment_cache_file
            else:
                self.sentiment_cache_file = os.path.join(base_dir, sentiment_cache_file)
        
        self.summary_cache_file = self.cache_file.replace('.json', '_summary.json')
        self.cache_ttl = 1800  # 30分钟缓存
        self.fingpt_analyzer = None
        self.qwen_optimizer = None
        
        self.sources = {
            "cryptonews": {
                "name": "CryptoNews",
                "url": "https://cryptonews.com/feed/bitcoin/",
                "enabled": True
            },
            "cointelegraph": {
                "name": "Cointelegraph",
                "url": "https://cointelegraph.com/rss/tag/bitcoin",
                "enabled": True
            },
            "coindesk": {
                "name": "CoinDesk",
                "url": "https://www.coindesk.com/arc/outboundfeeds/rss/category/bitcoin/",
                "enabled": True
            },
            "bitcoinmagazine": {
                "name": "Bitcoin Magazine",
                "url": "https://bitcoinmagazine.com/.rss/full/",
                "enabled": True
            },
            "binance": {
                "name": "币安公告",
                "url": "https://www.binance.com/bapi/composite/v1/public/cms/article/list/query",
                "enabled": True
            }
        }
    
    def set_fingpt_analyzer(self, analyzer):
        """设置 FinGPT 分析器"""
        self.fingpt_analyzer = analyzer
    
    def set_qwen_optimizer(self, optimizer):
        """设置 Qwen 优化器（用于翻译）"""
        self.qwen_optimizer = optimizer
    
    def _load_cache(self) -> List[Dict]:
        """从缓存加载新闻"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    last_update = datetime.fromisoformat(cache_data.get('last_update', ''))
                    if (datetime.now() - last_update).total_seconds() < self.cache_ttl:
                        return cache_data.get('news', [])
            except Exception:
                pass
        return []
    
    def _save_cache(self, news: List[Dict]):
        """保存新闻到缓存"""
        cache_data = {
            'last_update': datetime.now().isoformat(),
            'news': news
        }
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存新闻缓存失败: {e}")
    
    def _load_sentiment_cache(self) -> Dict:
        """从缓存加载情绪分析结果"""
        if os.path.exists(self.sentiment_cache_file):
            try:
                with open(self.sentiment_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_sentiment_cache(self, sentiment_data: Dict, keep_count: int = 20):
        """保存情绪分析结果到缓存（只保存最新的N条）"""
        try:
            # 只保留最新的 keep_count 条记录
            # 转换为列表并按时间戳排序（假设每条记录有 timestamp 字段）
            if sentiment_data:
                # 如果没有时间戳，添加当前时间戳
                for key, value in sentiment_data.items():
                    if 'timestamp' not in value:
                        value['timestamp'] = datetime.now().isoformat()
                
                # 按时间戳排序，只保留最新的 keep_count 条
                sorted_items = sorted(
                    sentiment_data.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )[:keep_count]
                
                # 重新构建字典
                sentiment_data = dict(sorted_items)
            
            with open(self.sentiment_cache_file, 'w', encoding='utf-8') as f:
                json.dump(sentiment_data, f, ensure_ascii=False, indent=2)
            
            print(f"[新闻爬虫] 情绪缓存已保存，共 {len(sentiment_data)} 条记录")
        except Exception as e:
            print(f"保存情绪分析缓存失败: {e}")
    
    def _load_summary_cache(self) -> Dict:
        """从缓存加载新闻总结"""
        if os.path.exists(self.summary_cache_file):
            try:
                with open(self.summary_cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def _save_summary_cache(self, summary_data: Dict, keep_count: int = 50):
        """保存新闻总结到缓存（只保存最新的N条）"""
        try:
            if summary_data:
                # 按时间戳排序，只保留最新的 keep_count 条
                sorted_items = sorted(
                    summary_data.items(),
                    key=lambda x: x[1].get('timestamp', ''),
                    reverse=True
                )[:keep_count]
                summary_data = dict(sorted_items)
            
            with open(self.summary_cache_file, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            print(f"保存新闻总结缓存失败: {e}")
    
    def _translate_title(self, title: str) -> str:
        """使用 Qwen 翻译新闻标题为中文（带缓存）"""
        if not self.qwen_optimizer:
            return title
        
        # 生成新闻唯一ID用于缓存
        news_id = f"title_{title[:100]}"
        
        # 检查缓存
        try:
            summary_cache = self._load_summary_cache()
            if news_id in summary_cache:
                cached_translation = summary_cache[news_id].get('translation', '')
                if cached_translation:
                    return cached_translation
        except Exception:
            pass
        
        try:
            print(f"[新闻爬虫] 正在翻译标题: {title[:50]}...")
            
            # 使用 Qwen 进行标题翻译 - 更严格的提示词
            prompt = f"""你是一个专业的翻译助手。请将以下英文新闻标题翻译成中文。
要求：
1. 只返回翻译结果
2. 不要添加任何其他文字、说明或对话
3. 翻译要准确、简洁，符合新闻标题风格

英文标题：
{title}

中文翻译："""
            
            response = self.qwen_optimizer.generate(prompt, max_tokens=80, temperature=0.1)
            
            if response:
                # 清理翻译结果
                translation = response.strip()
                
                # 移除可能的提示词残留
                translation = translation.replace('中文翻译：', '').replace('中文翻译:', '').strip()
                translation = translation.replace('翻译：', '').replace('翻译:', '').strip()
                
                # 移除可能的多余空行和空白
                lines = [line.strip() for line in translation.split('\n') if line.strip()]
                if lines:
                    translation = lines[0]
                
                # 只取前100字符，确保简洁
                translation = translation[:100].strip()
                
                # 验证翻译结果是否合理
                # 如果翻译结果包含很多英文或不相关内容，直接使用原标题
                if len(translation) < 5 or any(keyword in translation.lower() for keyword in ['好的', '请提供', '继续', '当然', '抱歉']):
                    translation = title
                
                # 如果清理后为空，使用原始标题
                if not translation:
                    translation = title
                
                # 保存到缓存
                try:
                    summary_cache = self._load_summary_cache()
                    summary_cache[news_id] = {
                        'translation': translation,
                        'timestamp': datetime.now().isoformat()
                    }
                    self._save_summary_cache(summary_cache)
                except Exception:
                    pass
                
                print(f"[新闻爬虫] 标题翻译完成")
                return translation
            
        except Exception as e:
            print(f"[新闻爬虫] 标题翻译失败: {e}")
        
        return title
    
    def _analyze_news_sentiment(self, news: Dict) -> Dict:
        """分析单条新闻的情绪"""
        if not self.fingpt_analyzer:
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0
            }
        
        try:
            text = f"{news['title']} {news['content']}"
            result = self.fingpt_analyzer.analyze_sentiment(text)
            
            sentiment = result.get('sentiment', 'neutral')
            score = result.get('score', 0.0)
            
            # 标准化情绪标签
            if sentiment in ['positive', 'bullish']:
                sentiment_label = 'positive'
            elif sentiment in ['negative', 'bearish']:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'sentiment': sentiment_label,
                'sentiment_score': score
            }
        except Exception as e:
            print(f"[新闻爬虫] 情绪分析失败: {e}")
            return {
                'sentiment': 'neutral',
                'sentiment_score': 0.0
            }
    
    def _parse_rss_feed(self, source_name: str, url: str) -> List[NewsItem]:
        """解析RSS源"""
        news_items = []
        try:
            print(f"[新闻爬虫] 正在从 {source_name} 获取新闻...")
            
            feed = feedparser.parse(url)
            
            for entry in feed.entries[:10]:
                title = entry.get('title', '')
                summary = entry.get('summary', '')
                
                # 只保留BTC相关新闻
                if 'bitcoin' not in title.lower() and 'btc' not in title.lower():
                    continue
                
                # 清理HTML标签
                soup = BeautifulSoup(summary, 'html.parser')
                clean_content = soup.get_text(strip=True)
                
                # 获取发布时间
                published_at = ''
                if hasattr(entry, 'published'):
                    try:
                        published_dt = datetime(*entry.published_parsed[:6])
                        published_at = published_dt.isoformat()
                    except:
                        published_at = datetime.now().isoformat()
                else:
                    published_at = datetime.now().isoformat()
                
                # 获取链接
                url = entry.get('link', '')
                
                # 尝试获取图片
                image_url = None
                if hasattr(entry, 'media_content'):
                    image_url = entry.media_content[0].get('url')
                
                news_item = NewsItem(
                    title=title,
                    content=clean_content[:200] + '...' if len(clean_content) > 200 else clean_content,
                    source=source_name,
                    url=url,
                    published_at=published_at,
                    image_url=image_url
                )
                news_items.append(news_item)
            
            print(f"[新闻爬虫] {source_name} 获取到 {len(news_items)} 条BTC新闻")
            
        except Exception as e:
            print(f"[新闻爬虫] {source_name} 获取失败: {e}")
        
        return news_items
    
    def _fetch_binance_news(self, source_name: str, url: str) -> List[NewsItem]:
        """获取币安公告"""
        news_items = []
        try:
            print(f"[新闻爬虫] 正在从 {source_name} 获取最新公告...")
            
            # 币安API请求参数
            params = {
                "pageNo": 1,
                "pageSize": 10,
                "catalogId": 48  # 48是币安公告目录
            }
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "application/json"
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data.get("code") == "000000" and data.get("data"):
                articles = data["data"].get("articles", [])
                count = 0
                
                for article in articles:
                    title = article.get("title", "")
                    content = article.get("mobileContent", "") or article.get("content", "")
                    
                    # 清理HTML标签
                    soup = BeautifulSoup(content, 'html.parser')
                    clean_content = soup.get_text(strip=True)
                    
                    # 获取发布时间
                    published_at = ''
                    if article.get("releaseDate"):
                        try:
                            published_ts = article["releaseDate"]
                            published_dt = datetime.fromtimestamp(published_ts / 1000)
                            published_at = published_dt.isoformat()
                        except:
                            published_at = datetime.now().isoformat()
                    else:
                        published_at = datetime.now().isoformat()
                    
                    # 构建链接
                    article_id = article.get("id", "")
                    article_url = f"https://www.binance.com/zh-CN/support/announcement/{article_id}" if article_id else ""
                    
                    news_item = NewsItem(
                        title=title,
                        content=clean_content[:200] + '...' if len(clean_content) > 200 else clean_content,
                        source=source_name,
                        url=article_url,
                        published_at=published_at
                    )
                    news_items.append(news_item)
                    count += 1
                    
                    # 只取最新3条
                    if count >= 3:
                        break
            
            print(f"[新闻爬虫] {source_name} 获取到 {len(news_items)} 条公告")
            
        except Exception as e:
            print(f"[新闻爬虫] {source_name} 获取失败: {e}")
        
        return news_items
    
    def fetch_all_news(self, force_refresh: bool = False, analyze_sentiment: bool = True, translate: bool = True) -> List[Dict]:
        """
        获取所有BTC新闻
        
        Args:
            force_refresh: 是否强制刷新
            analyze_sentiment: 是否分析情绪
            translate: 是否翻译为中文
            
        Returns:
            新闻列表
        """
        # 检查缓存
        if not force_refresh:
            cached_news = self._load_cache()
            if cached_news:
                print(f"[新闻爬虫] 使用缓存，共 {len(cached_news)} 条新闻")
                return cached_news
        
        # 获取新闻
        all_news = []
        
        for source_id, source_config in self.sources.items():
            if not source_config['enabled']:
                continue
            
            if source_id == "binance":
                news = self._fetch_binance_news(
                    source_config['name'],
                    source_config['url']
                )
            else:
                news = self._parse_rss_feed(
                    source_config['name'],
                    source_config['url']
                )
            all_news.extend(news)
        
        # 按时间排序（最新的在前）
        all_news.sort(
            key=lambda x: x.published_at,
            reverse=True
        )
        
        # 加载情绪缓存
        sentiment_cache = self._load_sentiment_cache()
        
        # 转换为字典并分析情绪
        news_dicts = []
        seen_titles = set()
        
        for item in all_news:
            if item.title in seen_titles:
                continue
            seen_titles.add(item.title)
            
            # 生成新闻唯一ID
            news_id = f"{item.source}_{item.title[:50]}"
            
            # 分析情绪
            sentiment_data = {'sentiment': 'neutral', 'sentiment_score': 0.0}
            if analyze_sentiment and self.fingpt_analyzer:
                if news_id in sentiment_cache:
                    sentiment_data = sentiment_cache[news_id]
                    print(f"[新闻爬虫] 使用缓存情绪分析: {item.title[:30]}...")
                else:
                    print(f"[新闻爬虫] 分析情绪: {item.title[:30]}...")
                    temp_news = {'title': item.title, 'content': item.content}
                    sentiment_data = self._analyze_news_sentiment(temp_news)
                    sentiment_data['timestamp'] = datetime.now().isoformat()
                    sentiment_cache[news_id] = sentiment_data
            
            # 翻译标题
            title_cn = None
            if translate and self.qwen_optimizer:
                title_cn = self._translate_title(item.title)
            
            news_dict = {
                'title': item.title,
                'title_cn': title_cn,
                'content': item.content,
                'source': item.source,
                'url': item.url,
                'published_at': item.published_at,
                'sentiment': sentiment_data['sentiment'],
                'sentiment_score': sentiment_data['sentiment_score'],
                'image_url': item.image_url
            }
            news_dicts.append(news_dict)
            
            if len(news_dicts) >= 30:
                break
        
        # 保存情绪缓存
        if analyze_sentiment:
            self._save_sentiment_cache(sentiment_cache)
        
        # 保存新闻缓存
        self._save_cache(news_dicts)
        
        print(f"[新闻爬虫] 共获取 {len(news_dicts)} 条高质量BTC新闻")
        return news_dicts


# 测试代码
if __name__ == "__main__":
    crawler = BTCNewsCrawler()
    news = crawler.fetch_all_news(force_refresh=True, analyze_sentiment=False, translate=False)
    
    print("\n" + "="*80)
    print("最新BTC新闻:")
    print("="*80)
    
    for i, item in enumerate(news[:5]):
        print(f"\n{i+1}. [{item['source']}] {item['title']}")
        print(f"   时间: {item['published_at']}")
        print(f"   链接: {item['url']}")
