import yaml

import torch
import streamlit as st

from model.model import Pegasus


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def init():
    model = Pegasus.load_from_checkpoint(
        "checkpoints/version_0/checkpoints/best.ckpt",
        hparams_file="checkpoints/version_0/hparams.yaml",
    )

    hparams = model.hparams
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = model.tokenizer
    model = model.model.to(device)

    return hparams, device, tokenizer, model


hparams, device, tokenizer, model = init()

context = st.text_area(
    "Context",
    """ If you own an iPhone 7 or 7 Plus, then you can easily restart it by pressing the correct buttons. In order to force reboot iPhone 6, you need to apply a different method, but to reboot an iPhone the ideal way, there is a simple technique. You can simply do it by pressing the power button.
Before we proceed and teach you how to restart iPhone, have a look at the anatomy of the device. The home button is located at the bottom while the volume up/down key is located on the left side. The Power (on/off or sleep/wake) button is located either on the right side or at the top.
Now, let’s proceed and learn how to reboot iPhone 7 and 7 Plus. You can do it by following these easy steps.
1. Start by pressing the Power (sleep/wake) button until a slider would appear on the screen.
2. Now, drag the slider to turn off your phone. Wait for a while as the phone vibrates and turns off.
3. When the device is switched off, hold the power button again until you see the Apple logo.
By following this drill, you would be able to restart your phone. Nevertheless, there are times when users need to force-restart their device. To force restart iPhone 7 or 7 Plus, follow these instructions.
1. Press the Power button on your device.
2. While holding the Power button, press the Volume down button.
3. Make sure that you keep holding both the buttons for another ten seconds. The screen will go blank and your phone will vibrate. Let go of them when the Apple logo appears on the screen. """,
)

question = st.text_area("Question", "How do I restart my phone?")

if st.button("Compute"):
    src = question + "\n" + context
    batch = tokenizer(
        src,
        truncation=True,
        max_length=hparams.max_input_length,
        return_tensors="pt",
    ).to(device)
    translated = model.generate(**batch)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    st.write(tgt_text[0])
