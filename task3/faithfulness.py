"""Manual claim-level faithfulness worksheet helper."""
import re
def split_claims(answer):
    return [x.strip() for x in re.split(r'(?<=[.!?])\s+',answer) if x.strip()]
def check_claims(answer, context):
    claims=split_claims(answer); lower=context.lower()
    return [{"claim":c,"supported":c.lower() in lower} for c in claims]
if __name__=='__main__':
    sample='Employees receive 12 casual leaves per year. The office is in Chennai.'
    context='Employees receive 12 casual leaves per calendar year.'
    for row in check_claims(sample,context): print(row)
