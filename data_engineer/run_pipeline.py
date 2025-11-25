import schedule
import time
from datetime import datetime

# Import ฟังก์ชันจากไฟล์ที่คุณทำไว้แล้ว
# (ต้องวางไฟล์ pipeline_fetching.py, pipeline_cleaning.py ไว้ที่เดียวกัน)
from pipeline_fetching import run_fetching_pipeline
from pipeline_cleaning import run_data_cleaning_pipeline

def job():
    print(f"⏰ Starting Job at {datetime.now()}")
    try:
        # Step 1: ดึงข้อมูล
        run_fetching_pipeline()

        # Step 2: คลีนข้อมูล
        run_data_cleaning_pipeline()

        print("✅ All tasks completed successfully!")
    except Exception as e:
        print(f"❌ Job Failed: {e}")

# ตั้งเวลา: เช่น รันทุกวันตอน 9 โมงเช้า
# schedule.every().day.at("09:00").do(job)

# หรือทดสอบ: รันทุกๆ 1 นาที (เพื่อดูผลทันที)
schedule.every(1).minutes.do(job)

print("🚀 Scheduler started... Waiting for time.")

# Loop เพื่อรอเวลาทำงาน
while True:
    schedule.run_pending()
    time.sleep(1)