error_types = ['mistake', 'slip', 'lapse']
idx_to_rule = {1: "hbt", 2: "ra", 3: "tb", 4: "oib"}
def error_category(human, rule_category, error_cat_label_mode = 'gt'):
    if error_cat_label_mode == 'gt':
        if rule_category == 'hbt':
            return [1, 0, 0]
        if rule_category == 'ra':
            return [1, 0, 0] 
        if rule_category == 'tb':
            return [0, 1, 0]
        if rule_category == 'oib':
            return [0, 0, 1]
    else:
        raise ValueError(f"Error category label mode {error_cat_label_mode} not implemented yet!")
    pass