#!/usr/bin/env python
"""Qwen 新闻预处理模块 - 语义理解、否定词检测、翻译为英文"""

import os
import sys
import logging
import json
import re
from typing import Dict, List, Optional
from datetime import datetime, timedelta

log_file = None
try:
    if getattr(sys, 'frozen', False):
        log_dir = os.path.join(os.path.dirname(sys.executable), 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, 'qwen_news_processor.log')
    
    if log_file:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename=log_file,
            filemode='a',
            encoding='utf-8'
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            stream=sys.stdout
        )
except Exception:
    pass

try:
    transformers_logger = logging.getLogger('transformers')
    transformers_logger.setLevel(logging.ERROR)
except Exception:
    pass

os.environ["HF_HUB_DISABLE_XET"] = "1"

try:
    from dotenv import load_dotenv
    if getattr(sys, 'frozen', False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.dirname(__file__)
    
    env_path = os.path.join(base_dir, ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    
    hf_token = os.getenv("HF_TOKEN")
    if hf_token and hf_token.strip() and not hf_token.startswith("your_"):
        os.environ["HF_TOKEN"] = hf_token
except ImportError:
    pass
except Exception:
    pass

QWEN_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    QWEN_AVAILABLE = True
except ImportError:
    print("警告: transformers库未安装，Qwen新闻处理器不可用")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: torch库未安装，Qwen新闻处理器不可用")


class QwenNewsProcessor:
    """Qwen 新闻处理器 - 语义理解、否定词检测、翻译为英文"""
    
    def __init__(self, 
                 model_path: str = None,
                 use_local_model: bool = True,
                 device: str = None,
                 cache_file: str = None):
        self.use_local_model = use_local_model
        self.model = None
        self.tokenizer = None
        self.available = False
        self.use_rule_based = False
        self.processed_cache = {}  # 缓存已处理的新闻
        
        if cache_file is None:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(__file__)
            cache_file = os.path.join(base_dir, "qwen_news_cache.json")
        
        self.cache_file = cache_file
        self._load_cache()
        
        if model_path is None:
            if getattr(sys, 'frozen', False):
                base_dir = os.path.dirname(sys.executable)
            else:
                base_dir = os.path.dirname(__file__)
            # 检查 _internal/models 目录（打包环境），如果不存在再检查 models 目录
            model_path = os.path.join(base_dir, "_internal", "models", "Qwen3.5-0.8B-Instruct")
            if not os.path.exists(model_path):
                model_path = os.path.join(base_dir, "models", "Qwen3.5-0.8B-Instruct")
        
        self.model_path = model_path
        
        if device is None:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        if QWEN_AVAILABLE and TORCH_AVAILABLE and use_local_model:
            self._initialize_model()
        else:
            #print("[Qwen新闻处理器] 使用规则-based 模式（模型不可用）")
            self.available = True
            self.use_rule_based = True
    
    def _initialize_model(self):
        try:
            print(f"[Qwen新闻处理器] 正在加载模型: {self.model_path}")
            
            if os.path.exists(self.model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_path, 
                    local_files_only=True,
                    trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    trust_remote_code=True,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device
                )
            else:
                #print(f"[Qwen新闻处理器] 本地模型不存在: {self.model_path}")
                #print("[Qwen新闻处理器] 使用规则-based 模式")
                self.available = True
                self.use_rule_based = True
                return
            
            self.available = True
            #print("[Qwen新闻处理器] 模型加载成功！")
            
        except Exception as e:
            print(f"[Qwen新闻处理器] 模型加载失败: {e}")
            print("[Qwen新闻处理器] 使用规则-based 模式")
            self.available = True
            self.use_rule_based = True
    
    def _load_cache(self):
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.processed_cache = json.load(f)
                #print(f"[Qwen新闻处理器] 已加载缓存: {len(self.processed_cache)} 条新闻")
        except Exception as e:
            #print(f"[Qwen新闻处理器] 加载缓存失败: {e}")
            self.processed_cache = {}
    
    def _save_cache(self):
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.processed_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            pass
    
    def _generate_with_qwen(self, prompt: str, max_new_tokens: int = 500) -> str:
        if self.use_rule_based or not self.model or not self.tokenizer:
            return None
        
        try:
            messages = [
                {"role": "system", "content": "你是一位专业的加密货币新闻分析专家。请准确、简洁地分析新闻内容，严格按照要求的JSON格式输出结果。不要添加任何额外的解释或说明。"},
                {"role": "user", "content": prompt}
            ]
            
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
            
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                top_p=0.9
            )
            
            generated_ids = [
                output_ids[len(input_ids):]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]
            
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            return response
        except Exception as e:
            return None
    
    def _rule_based_process(self, news: Dict) -> Dict:
        title = news.get("title", "")
        content = news.get("content", "")
        full_text = f"{title} {content}".lower()
        
        negation_words = ["not", "won't", "will not", "cancel", "postpone", "abandon", "reject", "deny", "refuse", "no", "never", "don't"]
        chinese_negations = ["不", "不会", "取消", "推迟", "放弃", "拒绝", "否认", "没有", "从未"]
        
        has_negation = any(neg in full_text for neg in negation_words)
        has_chinese_negation = any(neg in full_text for neg in chinese_negations)
        
        severe_keywords = ["ban", "crackdown", "lawsuit", "hack", "exploit", "breach", "rug pull", "flash crash", "crisis", "default", "bankrupt"]
        moderate_keywords = ["restrict", "cfdc", "attack", "vulnerability", "inflation", "recession", "bailout", "bug", "glitch", "downtime", "hard fork"]
        mild_keywords = ["maintenance", "update", "upgrade", "announcement", "partnership", "collaboration"]
        
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "cryptocurrency", "blockchain", "token", "exchange", "price", "market", "volume", "rally", "drop", "bull", "bear", "etf", "institutional", "adoption"]
        
        severe_count = sum(1 for kw in severe_keywords if kw in full_text)
        moderate_count = sum(1 for kw in moderate_keywords if kw in full_text)
        mild_count = sum(1 for kw in mild_keywords if kw in full_text)
        crypto_count = sum(1 for kw in crypto_keywords if kw in full_text)
        
        if severe_count > 0:
            severity = "SEVERE"
        elif moderate_count > 0:
            severity = "MODERATE"
        elif mild_count > 0 or crypto_count > 0:
            severity = "MILD"
        else:
            severity = "IRRELEVANT"
        
        published_at = news.get("published_at")
        is_recent = True
        if published_at:
            try:
                news_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                is_recent = (datetime.now() - news_time) < timedelta(hours=24)
            except:
                pass
        
        decision = "KEEP"
        if severity == "IRRELEVANT":
            decision = "FILTER"
        elif not is_recent:
            decision = "FILTER"
        elif (has_negation or has_chinese_negation) and severity == "MODERATE":
            decision = "FILTER"
        
        english_title = self._simple_translate(title)
        english_content = self._simple_translate(content)
        
        keywords = []
        if severe_count > 0:
            keywords.extend([kw for kw in severe_keywords if kw in full_text])
        if moderate_count > 0:
            keywords.extend([kw for kw in moderate_keywords if kw in full_text])
        
        return {
            "decision": decision,
            "severity": severity,
            "has_negation": has_negation or has_chinese_negation,
            "is_recent": is_recent,
            "reason": f"Rule-based: severity={severity}, recent={is_recent}, negation={has_negation or has_chinese_negation}",
            "keywords": keywords,
            "english_title": english_title,
            "english_content": english_content,
            "processed_at": datetime.now().isoformat()
        }
    
    def _simple_translate(self, text: str) -> str:
        chinese_to_english = {
            "比特币": "Bitcoin",
            "btc": "Bitcoin",
            "以太币": "Ethereum",
            "eth": "Ethereum",
            "加密货币": "cryptocurrency",
            "区块链": "blockchain",
            "交易所": "exchange",
            "监管": "regulation",
            "禁止": "ban",
            "黑客": "hack",
            "攻击": "attack",
            "崩溃": "crash",
            "下跌": "drop",
            "上涨": "rally",
            "市场": "market",
            "价格": "price",
            "交易量": "volume",
            "SEC": "SEC",
            "CFTC": "CFTC"
        }
        
        result = text
        for cn, en in chinese_to_english.items():
            result = result.replace(cn, en)
        
        return result
    
    def process_single_news(self, news: Dict) -> Dict:
        title = news.get("title", "")
        url = news.get("url", "")
        
        cache_key = f"{title}___{url}"
        
        if cache_key in self.processed_cache:
            #print(f"\n[Qwen新闻处理器] 缓存命中，跳过处理: {title[:50]}...")
            return self.processed_cache[cache_key]
        
        #print(f"\n[Qwen新闻处理器] 处理新闻: {title[:50]}...")
        
        if self.use_rule_based:
            result = self._rule_based_process(news)
            print(f"  [规则模式] 决策: {result['decision']}, 严重程度: {result['severity']}")
            self.processed_cache[cache_key] = result
            return result
        
        title = news.get("title", "")
        content = news.get("content", "")
        published_at = news.get("published_at", "")
        
        prompt = f"""请分析这篇加密货币新闻文章并提供结构化分析。

新闻标题: {title}
新闻内容: {content}
发布时间: {published_at}
当前时间 (UTC): {datetime.now().isoformat()}

请分析以下内容：
1. 否定词检测：新闻是否包含否定词，如"不"、"不会"、"取消"、"推迟"、"放弃"、"拒绝"、"否认"、"没有"、"从未"等（或英文等价词）？
2. 严重程度评估：
   - SEVERE（严重）：禁令、黑客攻击、漏洞利用、数据泄露、跑路、崩盘、闪崩、危机、违约、破产
   - MODERATE（中等）：监管讨论、限制、SEC/CFTC调查、攻击、漏洞、通胀、衰退、救助、漏洞、故障、停机、硬分叉
   - MILD（轻微）：常规维护、更新、合作、小型公告
   - IRRELEVANT（无关）：与加密货币市场影响无关
3. 时效性检查：这篇新闻是否在过去24小时内发布？
4. 翻译：将标题和内容翻译成英文（保留Bitcoin、Ethereum等加密货币术语不变）
5. 关键词：提取相关的金融关键词

仅输出JSON格式：
{{
    "decision": "KEEP" or "FILTER",
    "severity": "SEVERE" or "MODERATE" or "MILD" or "IRRELEVANT",
    "has_negation": true or false,
    "is_recent": true or false,
    "reason": "简要说明",
    "keywords": ["keyword1", "keyword2"],
    "english_title": "翻译后的标题",
    "english_content": "翻译后的内容"
}}"""
        
        response = self._generate_with_qwen(prompt, max_new_tokens=800)
        
        if response:
            try:
                json_match = re.search(r'\{[\s\S]*\}', response)
                if json_match:
                    result = json.loads(json_match.group())
                    result["processed_at"] = datetime.now().isoformat()
                    print(f"  [模型模式] 决策: {result.get('decision')}, 严重程度: {result.get('severity')}")
                    self.processed_cache[cache_key] = result
                    return result
            except Exception as e:
                pass
        
        #print(f"  [Qwen新闻处理器] 回退到规则模式")
        result = self._rule_based_process(news)
        self.processed_cache[cache_key] = result
        return result
    
    def process_news_batch(self, news_list: List[Dict]) -> List[Dict]:
        #print(f"\n[Qwen新闻处理器] 开始处理 {len(news_list)} 条新闻...")
        
        processed_news = []
        for news in news_list:
            result = self.process_single_news(news)
            
            if result.get("decision") == "KEEP":
                processed_news.append({
                    "original_title": news.get("title", ""),
                    "original_content": news.get("content", ""),
                    "title": result.get("english_title", news.get("title", "")),
                    "content": result.get("english_content", news.get("content", "")),
                    "source": news.get("source", ""),
                    "published_at": news.get("published_at", ""),
                    "severity": result.get("severity", "MILD"),
                    "keywords": result.get("keywords", []),
                    "qwen_analysis": result,
                    "url": news.get("url", "")
                })
        
        #print(f"[Qwen新闻处理器] 处理完成: 保留 {len(processed_news)}/{len(news_list)} 条新闻")
        self._save_cache()
        return processed_news


_qwen_processor_instance = None

def get_qwen_news_processor() -> Optional[QwenNewsProcessor]:
    global _qwen_processor_instance
    if _qwen_processor_instance is None:
        _qwen_processor_instance = QwenNewsProcessor()
    return _qwen_processor_instance


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen 新闻处理器测试")
    print("=" * 60)
    
    processor = QwenNewsProcessor(use_local_model=True)
    
    test_news = [
        {
            "title": "SEC will not ban Bitcoin",
            "content": "The SEC announced today that it will not impose a ban on Bitcoin trading.",
            "source": "CryptoNews",
            "published_at": (datetime.now() - timedelta(hours=2)).isoformat()
        },
        {
            "title": "Major exchange hacked, $100M lost",
            "content": "A major cryptocurrency exchange was hacked today, resulting in $100M losses.",
            "source": "SecurityAlert",
            "published_at": datetime.now().isoformat()
        },
        {
            "title": "Old news from last week",
            "content": "This is old news that should be filtered out.",
            "source": "OldNews",
            "published_at": (datetime.now() - timedelta(days=7)).isoformat()
        }
    ]
    
    results = processor.process_news_batch(test_news)
    
    print(f"\n{'=' * 60}")
    print(f"最终结果: 保留 {len(results)} 条新闻")
    print(f"{'=' * 60}")
