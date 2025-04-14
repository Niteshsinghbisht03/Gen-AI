from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict ,Annotated,Optional
from pydantic import BaseModel
load_dotenv()

model = ChatGoogleGenerativeAI(model = 'gemini-1.5-pro')

class Review(BaseModel):
    key_themes:Annotated[list[str],'write down all the key themes discussed in the review in a list']
    summary:Annotated[str,'a brief summary']
    sentiment:Annotated[str,'Return sentiment of review either negative , positive or neutral']
    pros:Annotated[Optional[list[str]],'write down all the pros inside a list']
    cons:Annotated[Optional[list[str]],'write down all the cons inside a list']
structured_model = model.with_structured_output(Review)

result = structured_model.invoke('''"I've been using the Xonic Pro 500 Wireless Noise-Cancelling Headphones for the past month, and I must say, these headphones have truly exceeded my expectations in more ways than one.

Sound Quality:
The audio performance is exceptional. The highs are crisp, the mids are balanced, and the bass is deep but not overwhelming. Whether I’m listening to classical music, EDM, or podcasts, the sound remains consistently immersive. The custom-tuned drivers make every note pop, and you can actually feel the depth in orchestral music or the punch in hip-hop tracks.

Noise Cancellation:
The active noise cancellation is impressive. I’ve used these during flights, at busy cafés, and even during my commute, and they significantly reduce background noise. It's not absolute silence like with some ultra-premium models, but it's enough that I often forget I'm in a noisy environment. There’s also a transparency mode that lets ambient sound through, which is perfect when I need to hear announcements or talk to someone without removing the headphones.

Battery Life:
Battery life is stellar — around 40 hours with ANC off and about 30 hours with ANC on. I only need to charge them once a week, even with heavy use. Fast charging is also a lifesaver: a 10-minute charge gives me almost 4 hours of playback.

Comfort and Design:
The build is sleek and minimalist, with plush memory foam ear cups that make them comfortable to wear for hours. The clamping force is just right — secure but not too tight. That said, in hot weather, the leatherette cushions can get a bit warm after extended use.

Connectivity and Features:
Pairing is instant via Bluetooth 5.3, and multipoint connectivity works flawlessly — I can seamlessly switch between my laptop and phone. There's also an app that allows EQ customization, firmware updates, and control over touch gestures, which is a nice bonus. Voice assistant support works fine, though I find myself rarely using it.

Mic Quality:
For calls, the mic is decent. It blocks out background noise and keeps my voice clear, though it's not studio-quality. Still, it works perfectly well for Zoom meetings and phone calls.

Final Thoughts:
If you're looking for a premium headphone experience without paying the ultra-premium price tag, the Xonic Pro 500 is a no-brainer. It hits the sweet spot between performance, features, and price. Whether you're a music lover, a remote worker, or just someone who wants peace and quiet, these headphones deliver.''')

print(result)