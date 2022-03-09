from global_workspace import GlobalWorkspace


class Agent:
    def __init__(self, num_perceptions, hidden_layer_units, evaluation_functions, preference_functions,
                 attentional_limit, adaptational_rate, maximum_happiness, happiness_rate,
                 error_threshold, error_activation_factor, happiness_thresholds):
        self.num_perceptions = num_perceptions
        self.hidden_layer_units = hidden_layer_units
        self.evaluation_functions = evaluation_functions
        self.preference_functions = preference_functions

        self.attentional_limit = attentional_limit
        self.maximum_happiness = maximum_happiness
        self.happiness = maximum_happiness
        self.happiness_rate = happiness_rate
        self.error_threshold = error_threshold
        self.error_activation_factor = error_activation_factor

        self.risk_threshold = happiness_thresholds['risk_threshold']
        self.death_threshold = happiness_thresholds['death_threshold']

        self.global_workspace = GlobalWorkspace(num_perceptions, adaptational_rate)

    def is_at_risk(self):
        return self.happiness <= self.risk_threshold

    def is_dead(self):
        return self.happiness <= self.death_threshold
