"""
Демонстрация работы сервиса.

Запуск:
    python demo.py                  # сервис уже запущен на localhost:8000
    python demo.py --url http://localhost:8000
    python demo.py --url http://localhost:8000 --key my_secret_key
"""

import argparse
import json
import sys

try:
    import httpx
except ImportError:
    import urllib.request as _req

BASE_URL = "http://localhost:8000"

# Первая реальная транзакция из датасета (легитимная)
LEGIT_TX = {
    "Time": 0.0, "Amount": 149.62,
    "V1": -1.359807, "V2": -0.072781, "V3":  2.536347, "V4":  1.378155,
    "V5": -0.338321, "V6":  0.462388, "V7":  0.239599, "V8":  0.098698,
    "V9":  0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
    "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
    "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
    "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
    "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
}

# Транзакция с аномальными признаками — высокая вероятность мошенничества
FRAUD_TX = {
    "Time": 406.0, "Amount": 0.0,
    "V1": -2.312227, "V2":  1.951992, "V3":  -1.609851, "V4":  3.997906,
    "V5": -0.522188, "V6": -1.426545, "V7":  -2.537387, "V8":  1.391657,
    "V9": -2.770089, "V10": -2.772272, "V11":  3.202033, "V12": -2.899907,
    "V13":  1.059594, "V14": -2.890083, "V15":  0.850738, "V16": -0.617710,
    "V17": -3.141940, "V18": -0.914505, "V19":  0.143940, "V20": 0.380219,
    "V21": 0.972168, "V22":  0.403088, "V23": -0.130721, "V24": 0.023297,
    "V25": 0.406629, "V26":  0.140169, "V27": 0.028631, "V28":  0.005539,
}


def _request(method: str, url: str, body=None, headers=None):
    try:
        import httpx
        r = httpx.request(method, url, json=body, headers=headers or {}, timeout=5)
        return r.status_code, r.json() if r.headers.get("content-type","").startswith("application/json") else r.text
    except ImportError:
        import urllib.request, json as _json
        data = _json.dumps(body).encode() if body else None
        req  = urllib.request.Request(url, data=data, headers={**(headers or {}), "Content-Type":"application/json"}, method=method)
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode()
            try: return resp.status, _json.loads(raw)
            except: return resp.status, raw


def main(base_url: str, api_key: str = ""):
    headers = {"x-api-key": api_key} if api_key else {}
    print(f"Сервис: {base_url}\n{'='*50}")

    # 1. Health check
    print("\n1. Health check")
    code, body = _request("GET", f"{base_url}/health")
    print(f"   GET /health → {code}")
    print(f"   {json.dumps(body, ensure_ascii=False, indent=3)}")

    # 2. Легитимная транзакция
    print("\n2. Легитимная транзакция (первая строка датасета)")
    code, body = _request("POST", f"{base_url}/predict", LEGIT_TX, headers)
    print(f"   POST /predict → {code}")
    print(f"   fraud_probability: {body.get('fraud_probability')}")
    print(f"   is_fraud:          {body.get('is_fraud')}")
    print(f"   threshold:         {body.get('threshold')}")

    # 3. Мошенническая транзакция
    print("\n3. Транзакция с аномальными признаками")
    code, body = _request("POST", f"{base_url}/predict", FRAUD_TX, headers)
    print(f"   POST /predict → {code}")
    print(f"   fraud_probability: {body.get('fraud_probability')}")
    print(f"   is_fraud:          {body.get('is_fraud')}")

    # 4. Метрики
    print("\n4. Prometheus-метрики (первые 5 строк)")
    code, body = _request("GET", f"{base_url}/metrics")
    if isinstance(body, str):
        for line in body.split("\n")[:8]:
            if line and not line.startswith("#"):
                print(f"   {line}")

    print(f"\n{'='*50}")
    print("Swagger UI: " + base_url + "/docs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=BASE_URL)
    parser.add_argument("--key", default="")
    args = parser.parse_args()
    try:
        main(args.url, args.key)
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("Убедитесь что сервис запущен: uvicorn src.service.main:app --reload")
        sys.exit(1)
