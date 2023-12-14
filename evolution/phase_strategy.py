strategy_dict = {
    "no change": 0,
    "rate": 1,
    "rate with one time change when fails": 2,
    "rate with continuous change when fails": 3,
    "continuous change when fails": 4
}

strategy_list = [
    "no change",
    "rate",
    "rate with one time change when fails",
    "rate with continuous change when fails",
    "continuous change when fails"
]

all_phase_list = [
    ["local", "global"],
    ["global", "local"]
]

class PhaseStrategy:
    def __init__(self, phase_list_idx, change_strategy, rate=0.5, max_evaluations=300):

        try:
            self.change_strategy = change_strategy # int
        except KeyError:
            self.change_strategy = 1
        
        self.phase_list = all_phase_list[phase_list_idx]
        self.rate = rate
        self.max_evaluations = max_evaluations
        self.cur_eval = 1
        self.cur_phase = None
        self.one_time_flag = False

        if rate is not None:
            self.change_flag = max_evaluations * rate
        else:
            self.change_flag = None

    def next(self, is_success=True):
        if self.cur_eval > self.max_evaluations:
            return None

        if self.change_strategy == 1:
            if self.cur_eval >= self.change_flag:
                self.cur_phase = self.phase_list[1]
            else:
                self.cur_phase = self.phase_list[0]
        
        elif self.change_strategy == 2:
            if self.one_time_flag:
                self.cur_phase = self.phase_list[0] if self.cur_phase == self.phase_list[1] else self.phase_list[1]
                self.one_time_flag = False
            elif is_success == False:
                print("Change phase")
                self.cur_phase = self.phase_list[0] if self.cur_phase == self.phase_list[1] else self.phase_list[1]
                self.one_time_flag = True
            else:
                if self.cur_eval > self.change_flag:
                    self.cur_phase = self.phase_list[1]
                else:
                    self.cur_phase = self.phase_list[0]
                self.one_time_flag = False
        
        elif self.change_strategy == 3:
            if is_success == False:
                print("change phase")
                self.cur_phase = self.phase_list[0] if self.cur_phase == self.phase_list[1] else self.phase_list[1]
            else:
                if self.cur_eval > self.change_flag:
                    self.cur_phase = self.phase_list[1]
                else:
                    self.cur_phase = self.phase_list[0]

        elif self.change_strategy == 4:
            if is_success == False:
                print("change phase")
                self.cur_phase = self.phase_list[0] if self.cur_phase == self.phase_list[1] else self.phase_list[1]
        
        self.cur_eval += 1
        return self.cur_phase
    
    def get_phases(self):
        return self.phase_list
        