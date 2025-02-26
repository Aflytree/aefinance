from apscheduler.schedulers.blocking import BlockingScheduler
import  efi_Backtesting


def job():
    efi_Backtesting.efi_backtesting()

scheduler = BlockingScheduler()
scheduler.add_job(job, 'cron', hour=9, minute=15)
scheduler.start()
