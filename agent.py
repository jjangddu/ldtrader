import numpy as np
import utils


class Agent:
    # 에이전트 상태가 구성하는 값 개수
    # ratio_hold, 주식 보유 비율
    # profitloss, 손익률
    # avg_buy_price, 주당 매수 단가
    STATE_DIM = 3

    # 행동
    ACTION_BUY = 0  # 매수
    ACTION_SELL = 1  # 매도
    ACTION_HOLD = 2  # 관망

    # 인공 신경망에서 확률을 구할 행동들
    ACTIONS = [ACTION_BUY, ACTION_SELL, ACTION_HOLD]
    NUM_ACTIONS = len(ACTIONS)

    def __init__(self, environment, initial_balance, min_trading_price, max_trading_price):
        # 현재 주식 가격을 가져오기 위한 environment
        self.environment = environment
        self.initial_balance = initial_balance

        # 최소 단일 매매 금액, 최대 단일 매매 금액
        self.min_trading_price = min_trading_price
        self.max_trading_price = max_trading_price

        # Agent 속성
        self.balance = initial_balance 
        self.num_stocks = 0  

        # 포트폴리오 가치: balance + num_stocks * 현재 주식 가격
        self.portfolio_value = 0
        self.num_buy = 0
        self.num_sell = 0 
        self.num_hold = 0

        # Agent 상태
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    # agent 초기화
    def reset(self):
        self.balance = self.initial_balance
        self.num_stocks = 0
        self.portfolio_value = self.initial_balance
        self.num_buy = 0
        self.num_sell = 0
        self.num_hold = 0
        self.ratio_hold = 0
        self.profitloss = 0
        self.avg_buy_price = 0

    # 초기 자본금 setting
    def set_balance(self, balance):
        self.initial_balance = balance

    # agent 상태 반환
    def get_states(self):
        # 주식 보유 비율
        self.ratio_hold = self.num_stocks * self.environment.get_price() \
            / self.portfolio_value
        
        # agent 상태 반환
        # 주식 보유 비율, 손익률, 주당 매수 단가 대비 주가 등락률
        return (
            self.ratio_hold,
            self.profitloss,
            (self.environment.get_price() / self.avg_buy_price) - 1 \
                if self.avg_buy_price > 0 else 0
        )

    def decide_action(self, pred_value, pred_policy, epsilon):
        # 확률이 가장 높은 행동의 확률
        confidence = 0.

        pred = pred_policy
        if pred is None:
            pred = pred_value

        # 예측 값이 없을 경우 탐험
        if pred is None:
            epsilon = 1
        # 값이 모두 같은 경우 탐험
        else:
            maxpred = np.max(pred)
            if (pred == maxpred).all():
                epsilon = 1

        # 탐험 결정
        if np.random.rand() < epsilon:
            exploration = True
            action = np.random.randint(self.NUM_ACTIONS)
        else:
            exploration = False
            action = np.argmax(pred)

        confidence = .5
        if pred_policy is not None:
            confidence = pred[action]
        elif pred_value is not None:
            confidence = utils.sigmoid(pred[action])

        return action, confidence, exploration

    def validate_action(self, action):
        # 매수 시, 적어도 1주를 살 수 있는지 확인
        if action == Agent.ACTION_BUY:
            if self.balance < self.environment.get_price():
                return False
        # 매도 시, 주식 잔고가 있는지 확인
        elif action == Agent.ACTION_SELL:
            if self.num_stocks <= 0:
                return False
        return True

    def decide_trading_unit(self, confidence):
        if np.isnan(confidence):
            return self.min_trading_price
        
        # confidence에 따라 min_trading_price + 추가적 거래 금액 계산
        added_trading_price = max(min(
            int(confidence * (self.max_trading_price - self.min_trading_price)),
            self.max_trading_price-self.min_trading_price), 0)
        trading_price = self.min_trading_price + added_trading_price
        return max(int(trading_price / self.environment.get_price()), 1)

    def act(self, action, confidence):
        # 매수 혹은 매도를 할 수 있는 지 확인
        if not self.validate_action(action):
            action = Agent.ACTION_HOLD

        # environment에서 현재 가격 얻기
        curr_price = self.environment.get_price()

        # 매수의 경우
        if action == Agent.ACTION_BUY:
            # 매수 수량 결정
            trading_unit = self.decide_trading_unit(confidence)
            balance = (
                self.balance - curr_price * trading_unit
            )
            # 보유 현금이 부족할 경우 보유 현금으로 최대한 매수
            if balance < 0:
                trading_unit = min(
                    int(self.balance / curr_price ),
                    int(self.max_trading_price / curr_price)
                )
            # agent 업데이트
            invest_amount = curr_price * trading_unit
            if invest_amount > 0:
                # 주당 매수 단가 갱신
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks + curr_price * trading_unit) \
                        / (self.num_stocks + trading_unit) 
                # 보유 현금 갱신
                self.balance -= invest_amount
                # 보유 주식 수 갱신
                self.num_stocks += trading_unit
                # 매수 횟수 증가
                self.num_buy += 1 

        # 매도의 경우
        elif action == Agent.ACTION_SELL:
            # 매도 수량 결정
            trading_unit = self.decide_trading_unit(confidence)
            # 보유 주식이 부족할 경우 최대한 매도
            trading_unit = min(trading_unit, self.num_stocks)

            # agent 업데이트
            invest_amount = curr_price * trading_unit
            if invest_amount > 0:
                # 주당 매수 단가 갱신
                self.avg_buy_price = \
                    (self.avg_buy_price * self.num_stocks - curr_price * trading_unit) \
                        / (self.num_stocks - trading_unit) \
                            if self.num_stocks > trading_unit else 0
                # 보유 현금을 갱신
                self.balance += invest_amount
                # 보유 주식 수를 갱신
                self.num_stocks -= trading_unit
                # 매도 횟수 증가
                self.num_sell += 1 

        # 관망
        elif action == Agent.ACTION_HOLD:
            self.num_hold += 1  # 관망 횟수 증가

        # 포트폴리오 가치 갱신
        self.portfolio_value = self.balance + curr_price * self.num_stocks
        self.profitloss = self.portfolio_value / self.initial_balance - 1
        return self.profitloss
