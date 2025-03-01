import smtplib
import email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

print(email)

def create_message(sender, receiver, subject, body):
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
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

def send(body):
    nbody = ""
    if type(body) == list:
        if len(body) != 0:
            nbody = '\n\n'.join(body)
        else:
            return
    else:
        nbody = body
    # 条件判断示例
    condition = True  # 这里可以根据实际情况修改条件
    if condition:
        receiver = '17301333257@163.com'
        sender = 'zhangaifei.2008@163.com'
        subject = '测试邮件'
        # body = '这是一个update更新：'
        smtp_server = 'smtp.163.com'  # 例如：smtp.gmail.com, smtp.office365.com等 smtp.16com smtp.163.com
        port = 465  # 例如：587 for Gmail, 25 for some others v465
        password = 'FHhPc9WARnuqsG2e'  # 这里应该是你的邮箱密码或应用专用密码，如果是Gmail，可能需要生成一个App密码
        message = create_message(sender, receiver, subject, nbody)
        send_email(smtp_server, port, sender, password, receiver, message)
    else:
        print("不满足发送条件，不发送邮件。")

        #FHhPc9WARnuqsG2e