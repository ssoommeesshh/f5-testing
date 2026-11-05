import re

def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9' ]", " ", text.lower())).strip()

def error_rates(reference: str, hypothesis: str) -> dict:
    r, h = normalize(reference), normalize(hypothesis)
    def ops(a, b):
        d = [[(i + j, 0, 0, 0) if not i or not j else None for j in range(len(b)+1)] for i in range(len(a)+1)]
        for i in range(1, len(a)+1): d[i][0] = (i, 0, i, 0)
        for j in range(1, len(b)+1): d[0][j] = (j, 0, 0, j)
        for i in range(1, len(a)+1):
            for j in range(1, len(b)+1):
                if a[i-1] == b[j-1]: d[i][j] = d[i-1][j-1]
                else:
                    candidates=[(d[i-1][j][0]+1, d[i-1][j][1],d[i-1][j][2]+1,d[i-1][j][3]),(d[i][j-1][0]+1,d[i][j-1][1],d[i][j-1][2],d[i][j-1][3]+1),(d[i-1][j-1][0]+1,d[i-1][j-1][1]+1,d[i-1][j-1][2],d[i-1][j-1][3])]
                    d[i][j] = min(candidates)
        return d[-1][-1]
    _, ws, wd, wi = ops(r.split(), h.split()); _, cs, cd, ci = ops(list(r), list(h))
    return {"normalized_reference": r, "normalized_hypothesis": h, "wer": (ws+wd+wi)/max(1,len(r.split())), "cer": (cs+cd+ci)/max(1,len(r)), "substitutions":ws, "deletions":wd, "insertions":wi}
