import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import logging
import pickle
import os


class RobustStockDataDownloader:
    def __init__(self, cache_dir="stock_cache", max_retries=5):
        self.cache_dir = cache_dir
        self.max_retries = max_retries
        self.session = self._create_robust_session()

        # 创建缓存目录
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _create_robust_session(self):
        """创建健壮的请求会话"""
        session = requests.Session()

        # 设置重试策略
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True
        )

        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        # 设置请求头
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Connection': 'keep-alive',
            'Referer': 'https://quote.eastmoney.com/'
        })

        return session

    def _get_cache_path(self, stock_code):
        """获取缓存文件路径"""
        return os.path.join(self.cache_dir, f"{stock_code}.pkl")

    def is_data_fresh(self, cache_path, max_age_hours=24):
        """检查缓存数据是否新鲜"""
        if not os.path.exists(cache_path):
            return False

        file_mtime = os.path.getmtime(cache_path)
        current_time = time.time()
        return (current_time - file_mtime) < (max_age_hours * 3600)

    def download_stock_data(self, stock_code, force_refresh=False, use_cache=True):
        """
        下载股票数据，支持缓存和重试
        """
        cache_path = self._get_cache_path(stock_code)

        # 检查缓存
        if use_cache and not force_refresh and os.path.exists(cache_path):
            if self.is_data_fresh(cache_path):
                try:
                    with open(cache_path, 'rb') as f:
                        data = pickle.load(f)
                    logging.info(f"使用缓存数据: {stock_code}")
                    return data
                except Exception as e:
                    logging.warning(f"读取缓存失败 {stock_code}: {e}")

        # 下载新数据
        data = self._download_with_retry(stock_code)

        # 保存到缓存
        if data is not None and use_cache:
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                logging.info(f"数据已缓存: {stock_code}")
            except Exception as e:
                logging.warning(f"缓存数据失败 {stock_code}: {e}")

        return data

    def _download_with_retry(self, stock_code):
        """带重试的数据下载"""
        url = self._build_url(stock_code)

        for attempt in range(self.max_retries + 1):
            try:
                logging.info(f"下载数据 {stock_code} (尝试 {attempt + 1}/{self.max_retries + 1})")

                response = self.session.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()

                # 检查数据是否有效
                if self._is_valid_data(data):
                    logging.info(f"数据下载成功: {stock_code}")
                    return data
                else:
                    logging.warning(f"数据无效: {stock_code}")
                    return None

            except requests.exceptions.RequestException as e:
                logging.warning(f"下载失败 {stock_code} (尝试 {attempt + 1}): {e}")

                if attempt < self.max_retries:
                    wait_time = 2 ** attempt  # 指数退避
                    logging.info(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    logging.error(f"下载失败 {stock_code}，已达到最大重试次数")
                    return None
            except Exception as e:
                logging.error(f"下载异常 {stock_code}: {e}")
                return None

        return None

    def _build_url(self, stock_code):
        """构建数据请求URL"""
        # 根据你的数据源调整URL
        if stock_code.startswith('6'):
            secid = f"1.{stock_code}"
        else:
            secid = f"0.{stock_code}"

        base_url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
        params = {
            'fields1': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'beg': '20230323',  # 可以根据需要调整
            'end': '20251025',
            'rtntype': '6',
            'secid': secid,
            'klt': '101',
            'fqt': '1'
        }

        return f"{base_url}?{self._encode_params(params)}"

    def _encode_params(self, params):
        """编码URL参数"""
        from urllib.parse import urlencode
        return urlencode(params)

    def _is_valid_data(self, data):
        """检查数据是否有效"""
        if not data or not isinstance(data, dict):
            return False

        # 根据实际API响应结构调整
        if data.get('rc') != 0:
            return False

        if not data.get('data'):
            return False

        return True

    def cleanup_old_cache(self, max_age_days=7):
        """清理过期缓存"""
        current_time = time.time()
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                filepath = os.path.join(self.cache_dir, filename)
                file_age = current_time - os.path.getmtime(filepath)

                if file_age > (max_age_days * 24 * 3600):
                    try:
                        os.remove(filepath)
                        logging.info(f"清理过期缓存: {filename}")
                    except Exception as e:
                        logging.warning(f"清理缓存失败 {filename}: {e}")