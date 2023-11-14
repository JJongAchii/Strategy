
import os
import sys
import datetime
import exchange_calendars as xcals
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import telepot
from telegram.ext import Updater
from telegram.ext import CommandHandler
from telegram.ext import MessageHandler, Filters

from email.message import EmailMessage
from smtplib import SMTP_SSL
from pathlib import Path

parent_parent_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_parent_folder not in sys.path:
    sys.path.append(parent_parent_folder)

from settings.monitoring import MonitoringConfig    # noqa
settings = MonitoringConfig()


class TelegramBot:
    def __init__(self):
        try:
            self.my_token = settings.code.at['telegram_token', 'code']
            self.chat_id = settings.code.at['telegram_chat_id', 'code']

            self.my_bot = telepot.Bot(self.my_token)
            self.updater = Updater(self.my_token)
            self.dispatcher = self.updater.dispatcher
        except:
            print("...emem")

    def send_message(self, msg='No Message'):
        self.my_bot.sendMessage(self.chat_id, msg)

    def send_photo(self, photo):
        self.my_bot.sendPhoto(self.chat_id, photo)

    def command_handler(self, cmd, func):
        self.dispatcher.add_handler(CommandHandler(cmd, func,))

    def message_handler(self, func):
        get_message_handler = MessageHandler(Filters.text & (~Filters.command), func)
        self.dispatcher.add_handler(get_message_handler)


def get_market_date(univ, start_date, end_date):
    """
    각 universe(미국: NYSE, 한국: KRX)에 따른 개장일

    Args:
        univ: (str) US or KR, Defaults to KR
        start_date: (str) from date
        end_date: (str) to date

    Returns: (list) start_date & end_date 사이의 시장 개장일

    """

    start_date = '2022-01-01' if not start_date else start_date
    end_date = '2022-12-31' if not end_date else end_date

    market_date = xcals.get_calendar('XNYS') if univ == 'US' else xcals.get_calendar('XKRX')
    market_date = market_date.schedule.loc[start_date:end_date]

    return market_date.index.astype('str').to_list()


def send_mail(to_mail, subject, content, files=False):
    """

    Args:
        to_mail: (list) 보낼 메일 주소들
        subject: (str) title
        content: (str) body
        files: (str) 첨부 파일

    Returns:
        (None)
    """
    msg = EmailMessage()

    msg['From'] = settings.mail_from
    msg['To'] = ', '.join(to_mail)
    msg['subject'] = subject

    msg.set_content(content)

    if files:
        for file in files:
            file_name = Path(file).name
            with open(file, 'rb') as f:
                msg.add_attachment(f.read(), maintype='application', subtype='octet-stream', filename=file_name)

    with SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(settings.mail_from, settings.code.at['mail_login', 'code'])
        smtp.send_message(msg)
        smtp.quit()


if __name__ == '__main__':
    mkt_date = get_market_date('KR', '2022-01-01', datetime.datetime.today().strftime('%Y-%m-%d'))
    message = 'Test\n' + '\n'.join(mkt_date[-10:])

    telegram = TelegramBot()
    telegram.send_message(message)
    send_mail(settings.mail_to, 'Test', message)
