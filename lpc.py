"""
Project Name    : 技術は使ってなんぼ！ for wealth
WEB URL         : https://tech-useit-wealth.com/
Creation Date   : 2023/8/20

Copyright © 2023 yonesuke. All rights reserved.

This source code or any portion thereof must not be
reproduced or used in any manner whatsoever.
"""

import parselmouth
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# 音声を読み込む
wav_name = "aiueo_zunda.wav"
snd = parselmouth.Sound(wav_name)

# フォルマント分析
formants_burg = snd.to_formant_burg(
    max_number_of_formants=5.0, maximum_formant=5500.0, window_length=0.025, pre_emphasis_from=50.0
)

# 各時刻における第1~第4フォルマントを取得する
formants = []
for t in formants_burg.xs():
    tmp = []
    for num in range(1, 5):
        tmp.append(formants_burg.get_value_at_time(formant_number=num, time=t, unit="HERTZ") / 1000)
    formants.append(tmp)
df = pd.DataFrame(formants, columns=["1", "2", "3", "4"], index=formants_burg.xs())
df = df[["1", "2"]]
df = df.dropna(how="any")
df = df.reset_index()

# k-meansクラスタリングを実行
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
kmeans.fit(df)
labels = kmeans.labels_
centers = kmeans.cluster_centers_
result_df = pd.concat([df, pd.DataFrame(labels, columns=["kmeans_result"])], axis=1)

colors = ["red", "blue", "green", "purple", "orange", "black"]

fig = plt.figure(figsize=(10, 8))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

print(result_df.head())
for i in range(n_clusters):
    ax1.scatter(result_df["1"][labels == i], result_df["2"][labels == i], label=i, color=colors[i], alpha=0.3)
ax2.scatter(result_df["index"], result_df["kmeans_result"])

ax1.set_title("k-means", size=14)
ax1.legend()
ax1.set_xlabel("folmant1", size=12)
ax1.set_ylabel("folmant2", size=12)

ax2.set_title("time-chart", size=14)
ax2.set_xlabel("time", size=12)
ax2.set_ylabel("cluster", size=12)
plt.savefig(f"output/{wav_name.split('.', 1)[0]}.png")
plt.show()
