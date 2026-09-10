from common import BaseStructuralDAGGenerator


class DAGGenerator(BaseStructuralDAGGenerator):
    planning_mode = "linear"
    length_mode = "hard"
    use_service_awareness = True
