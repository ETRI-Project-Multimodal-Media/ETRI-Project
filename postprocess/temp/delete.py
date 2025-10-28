TXT_DIR = "/home/kylee/kylee/LongVALE/logs/modality_split_0930.txt"
JSON_DIR = "/home/kylee/kylee/LongVALE/logs/sample.json"
AFTER_DIR = "/home/kylee/kylee/LongVALE/logs/sample_processed.json"
import ast
import json

input_file = TXT_DIR   # txt 파일 경로
output_file = JSON_DIR  # 결과 json 파일 경로
nopolicy_file = AFTER_DIR
# data_list = []

# with open(input_file, "r", encoding="utf-8") as f:
#     for line in f:
#         line = line.strip()
#         if not line:
#             continue
#         try:
#             # 문자열을 Python dict 로 변환
#             parsed = ast.literal_eval(line)
#             data_list.append(parsed)
#         except Exception as e:
#             print(f"파싱 실패: {e}\n문장: {line}")

# # JSON 파일로 저장
# with open(output_file, "w", encoding="utf-8") as f:
#     json.dump(data_list, f, ensure_ascii=False, indent=2)

# print(f"변환 완료: {output_file}")



with open(output_file, "r", encoding="utf-8") as f:
    data = json.load(f)   # 리스트 형태일 것임

for item in data:
    try:
        if "policy" in item["result"]:
            del item["result"]["policy"]
    except Exception as e:
        print(f"제거 실패: {e}")

with open(nopolicy_file, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"priority 제거 완료: {nopolicy_file}")
