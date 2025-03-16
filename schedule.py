from apscheduler.schedulers.blocking import BlockingScheduler
import efi_backtesting
import efi_email

import os

os.system("C:\\Users\\DELL\\PyCharmMiscProject\\.venv\\Scripts\\python.exe -m pip install --upgrade akshare")
efi_email.send("akshare update done")

def job():
    efi_backtesting.efi_backtesting()

scheduler = BlockingScheduler()
scheduler.add_job(job, 'cron', hour=9, minute=20)

scheduler.start()
