def analyze_experiment(report): return {"status":"deterministic_placeholder","keys":sorted(report)}
def rank_next_experiments(reports): return sorted(reports, key=lambda r:str(r.get("name","")))
