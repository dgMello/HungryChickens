from twilio.rest import Client

account_sid = None
auth_token = None
client = Client(account_sid, auth_token)

phone_nums = {
    'ethan': '+19787268796',
    'sahil': '+16179812337',
    'doug': '+15086544145',
    'sender': '+14176682787'
}

recipient = phone_nums['doug']
sender = phone_nums['sender']

bird_ref = {
    'american goldfinch': 'https://en.wikipedia.org/wiki/American_goldfinch',
    'black-capped chickadee': 'https://en.wikipedia.org/wiki/Black-capped_chickadee',
    'house sparrow': 'https://en.wikipedia.org/wiki/House_sparrow',
    'northern cardinal': 'https://en.wikipedia.org/wiki/Northern_cardinal',
    'northern mockingbird': 'https://en.wikipedia.org/wiki/Northern_mockingbird'
}

def msgSender(bird_type: str, send_text: bool):

        if bird_type in bird_ref:
            print("Bird type detected!")
            if send_text:
                message = client.messages.create(
                    to=recipient,
                    from_=sender,
                    body="Blink camera detected a " + bird_type + "! Read more about this bird at " + bird_ref[bird_type] + ".")
                print("sms: " + message.sid + " sent to: " + recipient)
            else:
                print("[mock text] Blink camera detected a " + bird_type + "! Read more about this bird at " + bird_ref[bird_type] + ".")
        else:
            print("Cannot detect bird type! Possibly not a bird.")






