from common import BaseStructuralDAGGenerator


class DAGGenerator(BaseStructuralDAGGenerator):
    planning_mode = "linear"
    length_mode = "soft"
    use_service_awareness = False
