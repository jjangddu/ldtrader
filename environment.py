class Environment:
    # 날짜, 시가, 고가, 저가, 종가 -> idx = 4
    PRICE_IDX = 4

    # 생성자
    def __init__(self, chart_data=None):
        self.chart_data = chart_data
        self.observation = None
        self.idx = -1

    # 환경 초기화
    def reset(self):
        self.observation = None
        self.idx = -1

    # 하루 씩 순회하며 chart data return (observation)
    def observe(self):
        if len(self.chart_data) > self.idx + 1:
            self.idx += 1
            self.observation = self.chart_data.iloc[self.idx]
            return self.observation
        return None

    # 종가 return
    def get_price(self):
        if self.observation is not None:
            return self.observation[self.PRICE_IDX]
        return None
