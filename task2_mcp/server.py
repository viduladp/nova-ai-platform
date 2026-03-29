

import json, time, datetime, jsonlines, random
from faker import Faker
from rapidfuzz import fuzz, process

fake = Faker()

with open("../nova_mock_db.json", "r") as f:
    db = json.load(f)

orders_index    = {o["order_id"]: o for o in db["orders"]}
customers_index = {c["customer_id"]: c for c in db["customers"]}
products_index  = {p["product_id"]: p for p in db["products"]}
faqs_list       = db["faqs"]

AUDIT_LOG_PATH = "../audit_log.jsonl"

def log_audit(tool_name, inputs, output, duration_ms):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tool": tool_name,
        "inputs": inputs,
        "output": output,
        "duration_ms": round(duration_ms, 2),
        "status": "error" if "error" in output else "success"
    }
    with jsonlines.open(AUDIT_LOG_PATH, mode="a") as writer:
        writer.write(entry)
    return entry

def get_order_status(order_id):
    start = time.time()
    order = orders_index.get(order_id)
    result = order if order else {"error": f"Order {order_id} not found"}
    log_audit("get_order_status", {"order_id": order_id}, result, (time.time()-start)*1000)
    return result

def initiate_return(order_id, reason):
    start = time.time()
    order = orders_index.get(order_id)
    if not order:
        result = {"error": f"Order {order_id} not found"}
    elif not order["return_eligible"]:
        result = {"error": "Return not eligible"}
    else:
        return_id = f"RET-{fake.bothify(text='####').upper()}"
        result = {"return_id": return_id, "status": "initiated",
                  "refund_timeline": "5-7 business days"}
    log_audit("initiate_return", {"order_id": order_id, "reason": reason}, result, (time.time()-start)*1000)
    return result

def get_product_info(product_id):
    start = time.time()
    product = products_index.get(product_id)
    result = product if product else {"error": f"Product {product_id} not found"}
    log_audit("get_product_info", {"product_id": product_id}, result, (time.time()-start)*1000)
    return result

def get_customer_profile(customer_id):
    start = time.time()
    customer = customers_index.get(customer_id)
    result = customer if customer else {"error": f"Customer {customer_id} not found"}
    log_audit("get_customer_profile", {"customer_id": customer_id}, result, (time.time()-start)*1000)
    return result

def search_faqs(query, top_k=3):
    start = time.time()
    questions = [f["question"] for f in faqs_list]
    matches = process.extract(query, questions, scorer=fuzz.WRatio, limit=top_k)
    results = [{"faq_id": faqs_list[i]["id"], "question": faqs_list[i]["question"],
                "answer": faqs_list[i]["answer"], "score": round(s/100,2)}
               for _, s, i in matches if s > 40]
    result = {"query": query, "results": results}
    log_audit("search_faqs", {"query": query}, result, (time.time()-start)*1000)
    return result
