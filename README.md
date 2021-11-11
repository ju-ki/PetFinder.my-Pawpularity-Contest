# PetFinder.my-Pawpularity-Contest

## OverView

一枚の絵は千の言葉に値する。しかし、1 枚の写真が 1,000 の命を救うことができることをご存知でしたか？世界中で毎日、何百万匹もの野良動物が路上で苦しんだり、保護施設で安楽死させられています。魅力的な写真を持つペットは、より多くの関心を集め、より早く採用されると思うかもしれません。しかし、良い写真とは何でしょうか？データサイエンスの助けを借りれば、ペットの写真の魅力を正確に判断し、救助された動物たちがより多くの人に愛されるチャンスを得られるよう、改善策を提案することができるかもしれません。

PetFinder.my は、マレーシアを代表する動物福祉プラットフォームで、18 万頭以上の動物が登録され、5 万 4 千頭が幸せな養子縁組をしています。ペットファインダーは、動物愛好家、メディア、企業、グローバル組織と密接に協力し、動物福祉の向上に努めています。

現在、PetFinder.my では、ペットの写真をランク付けするために、基本的な Cuteness Meter を使用しています。これは、写真の構図やその他の要素を分析し、何千ものペットのプロフィールのパフォーマンスと比較するものです。この基本的なツールは有用ですが、まだ実験的な段階であり、アルゴリズムは改善できる可能性があります。

このコンペティションでは、生の画像とメタデータを分析して、ペットの写真の「Pawpularity」を予測します。このモデルは、PetFinder.my の何千ものペットのプロフィールを使ってトレーニングとテストを行います。採用されたモデルは、動物福祉を向上させるための正確な推奨事項を提供します。

成功すれば、世界中のシェルターやレスキュー隊がペットのプロフィールの魅力を向上させるための AI ツールに採用され、自動的に写真の品質を向上させたり、構図の改善を推奨したりすることになります。その結果、野良犬や野良猫がより早く「運命の人」を見つけることができるようになります。Kaggle コミュニティのちょっとした支援で、多くの尊い命が救われ、より多くの幸せな家族が生まれるかもしれません。

上位の参加者には、ソリューションの実装に向けた共同作業に招待される可能性もあり、AI のスキルで世界の動物福祉を創造的に改善することができます。

## Rules

- deadline => 2022/01/14
- Evaluation => RMSE
- CPU Notebook <= 9 hours run-time
- GPU Notebook <= 9 hours run-time
- Internet access disabled
- Freely & publicly available external data is allowed, including pre-trained models
- Submission file must be named submission.csv

### 20211109

- petfinder 参戦, 目標は現時点での画像系の知識の習得+銀メダル以上
- とりあえず nakama さんのノートブックを自分用に改良
- 右も左も分からないので公開ノートブックをベースに頑張る

### 20211110

- transformer > CNN
- classification > regression
- transformer -> SwinTransformer
- phalanx さんの github を参考にどのように取り組むかを把握(したつもり)
- とりあえず Stratified 5fold, epoch:20, lr:1e-4, batch_size:32, optimizer:AdamW, scheduler:CosineAnnealingLR, seed:42 で固定
- grad_cam?(CNN で可視化させるやつができない)

### 20211111

- petfinder_baseline(tf_efficientnet) cv:19.3128 LB:19.11319
- exp1 swin_transformer
- どうやら画像に重複があるらしい(またはかなり似ている) -> 除去後のモデルもありかも
