"""Timer"""
import time


class Timer:
    """记录每一步执行时间的类"""

    def __init__(self):
        self.flag = False
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0
        self.curr_desc = ""
        self.map = {}
        self.map_time = {}

    def start(self, desc=""):
        """开始计时"""
        self.flag = True
        self.curr_desc = desc
        self.start_time = time.time()
        if desc not in self.map:
            self.map[desc] = 0
            self.map_time[desc] = 0

    def next(self, desc=""):
        """记录上一步的时间并开始计时"""
        if self.flag:
            self.end_time = time.time()
            print(f"{self.curr_desc} time: {self.end_time - self.start_time}s")
            self.map[self.curr_desc] += self.end_time - self.start_time
            self.map_time[self.curr_desc] += 1
            self.curr_desc = desc
            if desc not in self.map:
                self.map[desc] = 0
                self.map_time[desc] = 0
            self.total_time += self.end_time - self.start_time
            self.start_time = self.end_time
            return self.total_time
        else:
            return 0

    def end(self):
        """结束计时"""
        self.end_time = time.time()
        print(f"{self.curr_desc} time: {self.end_time - self.start_time}s")
        self.map[self.curr_desc] += self.end_time - self.start_time
        self.map_time[self.curr_desc] += 1
        self.total_time += self.end_time - self.start_time

    def reset(self):
        """重置计时器"""
        self.start_time = 0
        self.end_time = 0
        self.total_time = 0

    def get_time(self):
        """获取总时间"""
        return self.total_time

    def print_map(self):
        """打印时间表"""
        print("Total time:", self.total_time)
        max_key_length = max(len(str(k)) for k in self.map.keys())

        for k, v in self.map.items():
            v_ = f"{v:.3f}"
            ratio = f"{(v / self.total_time) * 100:.3f}%"
            padded_k = f"{k:{chr(12288)}<{max_key_length}}"
            print(f"{padded_k} {v_}s (ratio: {ratio}) (count: {self.map_time[k]})")

