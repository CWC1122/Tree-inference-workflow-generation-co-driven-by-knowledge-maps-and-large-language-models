from common import BaseStructuralDAGGenerator


class DAGGenerator(BaseStructuralDAGGenerator):
    planning_mode = "linear"
    length_mode = "none"
    use_service_awareness = True
