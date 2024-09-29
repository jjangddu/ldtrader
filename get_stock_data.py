import yfinance as yf
import json
import os

# csvfolder 경로 설정
folder_path = 'csvfolder'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)  # 폴더가 없으면 생성

# 종목 티커
ticker = "AAPL"  # 애플 주식 예시
stock = yf.Ticker(ticker)

data = stock.history(period="1y",actions=True)

# 데이터프레임을 csv로 저장
data.to_csv(os.path.join(folder_path, f"{ticker}.csv"))

# 종목의 상세 정보 가져오기
info = stock.info

# info 데이터를 json 형식으로 저장할 준비
info_str = json.dumps(info, indent=4)

# 파일 경로 설정 (csvfolder 안에 저장)
file_path = os.path.join(folder_path, f"{ticker}_info.txt")

# 파일에 저장
with open(file_path, 'w') as f:
    f.write(info_str)

print(f"{ticker} 정보가 {file_path}에 저장되었습니다.")