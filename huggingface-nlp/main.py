# https://www.meti.go.jp/policy/mono_info_service/geniac/geniac_magazine/resultsreport_2.html
text = """社会実装が続々登場！ GENIAC第2期採択事業者成果報告会を開催しました！
2025年6月9日（月）、東京・九段会館において、第2期GENIACで生成AI基盤モデル開発の支援事業に採択された19の事業者による、成果報告会が開催されました。第１部では、採択事業者による開発およびビジネスの進捗や成果、計算資源の提供側であるハイパースケーラーなどが総括コメントを発表。第2部では高度なモデルや社会実装などを評価する表彰式が催されました。イベント当日の模様をご紹介します。
【第1部】第2期GENIAC採択事業者による成果報告とハイパースケーラーによる総括講演
はじめに、経済産業省 商務情報政策局 情報処理基盤産業室 室長 渡辺 琢也が、GENIAC第2期の半年間の取り組みを振り返り、今後の展望などを話しました。

次に、今回の19社が達成した成果について発表しました。
株式会社ABEJA
株式会社AIdeaLab
AiHUB株式会社
AI inside株式会社
株式会社EQUES
株式会社Kotoba Technologies Japan
NABLAS株式会社
株式会社Preferred Elements/株式会社Preferred Networks
SyntheticGestalt株式会社
Turing株式会社
ウーブン・バイ・トヨタ株式会社
国立研究開発法人海洋研究開発機構
カラクリ株式会社
ストックマーク株式会社
株式会社データグリッド
株式会社ヒューマノーム研究所
フューチャー株式会社
株式会社リコー
株式会社ユビタス/株式会社Deepreneur
ハイパースケーラーによる第２サイクルの総括と期待の声
採択事業者のプレゼンテーションの後、計算資源を提供したハイパースケーラー、関係者による総括コメントが寄せられました。

アマゾンウェブサービスジャパン合同会社 常務執行役員 サービス&テクノロジー統括本部 統括本部長 安田 俊彦氏
"""

from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# model_name = "Mizuiro-sakura/luke-japanese-base-finetuned-ner" # Accuracy is lower than xlm-roberta-ner-japanes
model_name = "tsmatz/xlm-roberta-ner-japanese"
# 上記はStockmark "Wikipediaを用いた日本語の固有表現抽出データセット"を使ったfine-tuningモデル https://github.com/stockmarkteam/ner-wikipedia-dataset

# model_name = "ku-nlp/deberta-v3-base-japanese" # not for NER

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForTokenClassification.from_pretrained(model_name)

ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple",
)

results = ner(text)

for r in results:
    print(f"{r['word']} ({r['entity_group']}): score={r['score']:.4f}")


# Label id	Tag	Tag in Widget	Description
# 0	O	(None)	others or nothing
# 1	PER	PER	person
# 2	ORG	ORG	general corporation organization
# 3	ORG-P	P	political organization
# 4	ORG-O	O	other organization
# 5	LOC	LOC	location
# 6	INS	INS	institution, facility
# 7	PRD	PRD	product
# 8	EVT	EVT	event
