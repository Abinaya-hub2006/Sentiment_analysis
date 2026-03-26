import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def generate_charts(df):
    os.makedirs("outputs/images", exist_ok=True)

    counts = df['Sentiment'].value_counts()

    plt.figure()
    counts.plot.pie(autopct='%1.1f%%')
    plt.savefig("outputs/images/pie.png")
    plt.close()

    plt.figure()
    counts.plot.bar()
    plt.savefig("outputs/images/bar.png")
    plt.close()

    text = " ".join(df['Text'])
    wc = WordCloud().generate(text)

    plt.figure()
    plt.imshow(wc)
    plt.axis("off")
    plt.savefig("outputs/images/wordcloud.png")
    plt.close()