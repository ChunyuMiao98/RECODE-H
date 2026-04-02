from pyccmetrics import Metrics

metrics = Metrics("/home/linus/code/research_code_gen/code_agent/work_dir/dataset/annotations/annotation_3/canonical.py")

metrics.calculate()

print(metrics.metrics_dict)