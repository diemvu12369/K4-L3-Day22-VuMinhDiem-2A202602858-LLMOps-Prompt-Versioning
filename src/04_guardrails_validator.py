"""
Bước 4 — Guardrails AI Validators
====================================
NHIỆM VỤ:
  1. Xây dựng PIIDetector: phát hiện & redact email, số điện thoại, SSN, số thẻ tín dụng
  2. Xây dựng JSONFormatter: tự động sửa JSON lỗi
  3. Bọc mỗi validator trong Guard và test với các mẫu đầu vào
  4. Chạy demo với các trường hợp PII và JSON

DELIVERABLE: Tất cả test cases pass (PII bị redact, JSON được sửa thành công)

CÁC KHÁI NIỆM CHÍNH:
  - @register_validator     — khai báo custom validator class
  - Validator.validate()    — implement logic kiểm tra + sửa
  - OnFailAction.FIX        — thay thế output thay vì raise error
  - Guard().use(validator)  — gắn validator instance vào guard
  - guard.validate(text)    → ValidationOutcome
      .validation_passed    — bool
      .validated_output     — output đã được xử lý

⚠️  QUAN TRỌNG: on_fail phải truyền vào CONSTRUCTOR của VALIDATOR, KHÔNG phải Guard.use()
    SAI  : Guard().use(PIIDetector, on_fail=OnFailAction.FIX)   ← TypeError
    ĐÚNG : Guard().use(PIIDetector(on_fail=OnFailAction.FIX))   ← correct

Cách chạy:
    python 04_guardrails_validator.py              # cả 2 demo
    python 04_guardrails_validator.py --demo pii   # chỉ demo PII
    python 04_guardrails_validator.py --demo json  # chỉ demo JSON
"""

import re
import sys
import json

from guardrails import Guard
from guardrails.validators import Validator, register_validator, PassResult, FailResult

try:
    from guardrails.hub import OnFailAction
except ImportError:
    from guardrails.validator_base import OnFailAction


# ── 1. PII Detector Validator ──────────────────────────────────────────────
@register_validator(name="custom/pii-detector", data_type="string")
class PIIDetector(Validator):
    """
    Phát hiện và redact Personally Identifiable Information (PII) bằng regex.

    Các pattern được phát hiện:
      EMAIL       : xxx@xxx.xxx
      CREDIT_CARD : 1234 5678 9012 3456 (hoặc dấu gạch nối)
      SSN         : 123-45-6789
      PHONE       : (123) 456-7890 hoặc 123-456-7890

    Thứ tự có chủ đích: các pattern dài/cụ thể (thẻ, SSN) chạy trước PHONE, và mỗi
    pattern chạy trên text ĐÃ redact → một chuỗi số không bị gắn 2 nhãn khác nhau.
    """

    PII_PATTERNS = {
        "EMAIL":       r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "SSN":         r"\b\d{3}-\d{2}-\d{4}\b",
        "PHONE":       r"(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\b\d{3})[-.\s]\d{3}[-.\s]\d{4}\b",
    }

    def validate(self, value: str, metadata: dict):
        """
        Tìm PII trong value; nếu phát hiện, redact và trả về FailResult với
        fix_value là text đã xử lý.

        ⚠️ Với OnFailAction.FIX, Guardrails CHỈ thay output bằng FailResult.fix_value.
           PassResult(value_override=...) KHÔNG có tác dụng → output giống hệt input.
        """
        redacted_text = value
        found_pii     = []

        for pii_type, pattern in self.PII_PATTERNS.items():
            for match in re.findall(pattern, redacted_text):
                redacted_text = redacted_text.replace(match, f"[{pii_type}_REDACTED]")
                found_pii.append((pii_type, match))

        if found_pii:
            print(f"  ⚠️  Đã redact {len(found_pii)} PII: {[p[0] for p in found_pii]}")
            return FailResult(error_message="Phát hiện PII", fix_value=redacted_text)

        return PassResult()


# ── 2. JSON Formatter Validator ────────────────────────────────────────────
@register_validator(name="custom/json-formatter", data_type="string")
class JSONFormatter(Validator):
    """
    Validate và tự động sửa JSON lỗi.

    Các lỗi có thể sửa tự động:
      - Strip markdown code fences (``` hoặc ```json)
      - Thay single quotes → double quotes
      - Xóa trailing commas trước } hoặc ]
      - Re-serialize với json.dumps để định dạng chuẩn
    """

    @staticmethod
    def _repair(text: str) -> str:
        """
        Cố gắng sửa chuỗi JSON lỗi (chưa re-serialize).

        Lưu ý: thay nháy đơn là heuristic — sẽ làm hỏng chuỗi chứa dấu nháy
        (vd. "it's"); khi đó validate() rơi về JSON dự phòng.
        """
        text = text.strip()

        # Xóa markdown fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$',          '', text)
        text = text.strip()

        # Nháy đơn kiểu Python dict → nháy đôi chuẩn JSON
        text = text.replace("'", '"')

        # Xóa trailing commas trước } hoặc ]
        text = re.sub(r',\s*([}\]])', r'\1', text)

        return text

    def validate(self, value: str, metadata: dict):
        """
        Thử parse value thành JSON; nếu thất bại, gọi _repair() rồi thử lại.

        - JSON hợp lệ sẵn          → PassResult()
        - Sửa được                 → FailResult(fix_value=JSON đã chuẩn hoá)
        - Không sửa được           → FailResult(fix_value=<JSON dự phòng>)

        ⚠️ Với OnFailAction.FIX, chỉ FailResult.fix_value mới thay được output.
        """
        try:
            json.loads(value)
            return PassResult()
        except json.JSONDecodeError:
            pass

        try:
            repaired_text = self._repair(value)
            parsed        = json.loads(repaired_text)
            print(f"  🔧 JSON đã được sửa thành công")
            return FailResult(error_message="JSON lỗi, đã tự sửa", fix_value=json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
            # Không sửa được → trả về JSON dự phòng để output vẫn là JSON hợp lệ
            fallback = json.dumps({"error": "Không thể phân tích JSON", "raw": value[:200]}, ensure_ascii=False)
            return FailResult(error_message="Không thể sửa JSON", fix_value=fallback)


# ── 3. Demo: PII Guard ─────────────────────────────────────────────────────
def demo_pii_guard():
    print("\n" + "=" * 55)
    print("  Demo: PII Detection & Redaction")
    print("=" * 55)

    guard = Guard().use(PIIDetector(on_fail=OnFailAction.FIX))

    # Dữ liệu PII giả (RULES.md §6)
    test_cases = [
        ("Email",        "Contact John at john.doe@example.com for details."),
        ("Phone",        "Call our support line at (555) 867-5309."),
        ("SSN",          "Patient SSN is 123-45-6789 on file."),
        ("Credit Card",  "Payment made with card 4532 1234 5678 9010."),
        ("Multi-PII",    "Email: alice@example.com, Phone: 555-123-4567"),
        ("LLM answer",   "Sure! Reach Bob at bob.smith@test.org or 555.222.3333; card 4111-1111-1111-1111."),
        ("Clean",        "No sensitive information in this text."),
    ]

    for label, text in test_cases:
        result = guard.validate(text)

        print(f"\n[{label}]")
        print(f"  Input:  {text}")
        print(f"  Output: {result.validated_output}")


# ── 4. Demo: JSON Guard ────────────────────────────────────────────────────
def demo_json_guard():
    print("\n" + "=" * 55)
    print("  Demo: JSON Formatting & Repair")
    print("=" * 55)

    guard = Guard().use(JSONFormatter(on_fail=OnFailAction.FIX))

    test_cases = [
        ("Valid JSON",       '{"name": "Alice", "age": 30}'),
        ("Markdown fences",  '```json\n{"name": "Bob"}\n```'),
        ("Single quotes",    "{'name': 'Charlie', 'score': 95}"),
        ("Trailing comma",   '{"key": "value",}'),
        ("Combined errors",  "```json\n{'items': ['a', 'b',], 'ok': true,}\n```"),
        ("Truly invalid",    "This is not JSON at all: ??? {]"),
    ]

    for label, text in test_cases:
        result = guard.validate(text)

        # Guardrails 0.11 báo validation_passed=True cả khi đã FIX → phân loại theo output.
        output = str(result.validated_output)
        if output == text:
            status = "✅ Valid (giữ nguyên)"
        elif output.startswith('{"error"'):
            status = "⚠️ Fallback (không sửa được)"
        else:
            status = "🔧 Repaired"
        print(f"\n[{label}] {status}")
        print(f"  Input:  {text!r}"[:80])
        print(f"  Output: {output}")


# ── 5. Main ────────────────────────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  Bước 4: Guardrails AI Validators")
    print("=" * 55)

    # Cho phép chạy riêng từng demo để lưu 2 file evidence riêng biệt.
    demo = None
    if "--demo" in sys.argv:
        idx = sys.argv.index("--demo")
        demo = sys.argv[idx + 1] if idx + 1 < len(sys.argv) else None

    if demo in (None, "pii"):
        demo_pii_guard()
    if demo in (None, "json"):
        demo_json_guard()

    print("\n✅ Bước 4 hoàn thành!")


if __name__ == "__main__":
    main()
