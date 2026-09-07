import smtplib
import time
import logging
from pathlib import Path

from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def create_message(sender, receiver, subject, body, attachments=None):
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = receiver
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    for p in attachments or []:
        path = Path(p)
        if not path.is_file():
            continue
        with open(path, "r", encoding="utf-8") as f:
            part = MIMEText(f.read(), "plain", "utf-8")
        part.add_header("Content-Disposition", "attachment", filename=path.name)
        msg.attach(part)
    return msg.as_string()

def send_email(smtp_server, port, sender_email, password, receiver_email, message):
    try:
        server = smtplib.SMTP_SSL(smtp_server, port)
        # server.starttls()  # 启用TLS安全传输
        server.set_debuglevel(1)  # 开启调试模式以查看详细信息
        server.login(sender_email, password)
        # server.starttls(timeout=30)  # 增加超时时间到30秒
        server.sendmail(sender_email, receiver_email, message)
        server.quit()
        print("邮件发送成功！")
    except Exception as e:
        print(f"发送邮件时出错: {e}")

def send(body, attachments=None, subject=None):
    nbody = ""
    if type(body) == list:
        if len(body) != 0:
            nbody = "\n\n".join(body)
        else:
            return
    else:
        nbody = body
    # 条件判断示例
    condition = True  # 这里可以根据实际情况修改条件
    if condition:
        # receiver = '17301333257@163.com'
        receiver = "19282286879@163.com"
        sender = "zhangaifei.2008@163.com"
        subject = subject or "test case"
        smtp_server = "smtp.163.com"
        port = 465
        password = "FHhPc9WARnuqsG2e"
        message = create_message(sender, receiver, subject, nbody, attachments=attachments)
        send_email(smtp_server, port, sender, password, receiver, message)
    else:
        print("不满足发送条件，不发送邮件。")


def build_backtest_result_mail(
    stock_codes,
    all_results,
    signal_day,
    freshness_lines,
    one_d_list,
    last_buys_list,
    pattern_lines,
):
    """组装回测结果邮件正文行。"""
    mail_lines = [
        f"回测运行日: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"股票数: {len(stock_codes)}, 有效回测: {len(all_results)}",
        "",
    ]
    mail_lines.extend(freshness_lines)
    mail_lines.extend([
        "",
        "--------------------------------------",
        f" signal-day trades ({signal_day})",
        "--------------------------------------",
    ])
    if one_d_list:
        mail_lines.extend(one_d_list)
    else:
        mail_lines.append(f"(无 {signal_day} 买卖)")
    mail_lines.extend([
        "",
        "--------------------------------------",
        " current holds",
        "--------------------------------------",
    ])
    if last_buys_list:
        mail_lines.extend(last_buys_list)
    else:
        mail_lines.append("(无当前持仓)")
    mail_lines.extend(pattern_lines)
    return mail_lines


def send_backtest_result_mail(
    stock_codes,
    all_results,
    signal_day,
    freshness_lines,
    one_d_list,
    last_buys_list,
    pattern_lines,
    subject="test",
):
    """组装回测结果邮件、打持仓日志并发送。"""
    mail_lines = build_backtest_result_mail(
        stock_codes,
        all_results,
        signal_day,
        freshness_lines,
        one_d_list,
        last_buys_list,
        pattern_lines,
    )
    logging.info(
        "信号日买卖 %s 条, 当前持仓 %s 条",
        len(one_d_list),
        len(last_buys_list),
    )
    for line in last_buys_list:
        logging.info("%s", line)
    send(mail_lines, subject=subject)
    return mail_lines
