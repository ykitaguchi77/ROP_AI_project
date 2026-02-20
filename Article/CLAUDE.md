論文化のための叩き台です。自分はAI解析担当なので、このコーナーでは、material and methodsとResultsの部分に限定して執筆を行います。科学論文形式、英語です。
原稿はarticle.mdに記載します。

記載したい内容
・lens detectとsegmentation(disc, retina, macula)の2段階であること、また用いたモデル
・maculaは精度が低かったため解析からは外し、discとretinaを最適画像選出に用いたこと
・最適画像選出のプロトコルとその結果（human bestとの比較）
・次に、別のデータセット（多施設動画）についてのstage, zone, treatment, APROP, RWROP etc判定モデルの構築について
・それぞれの項目の定義
・判定モデル構築にあたり、Good/fair/bad/worstを手動分類し、goodとfairのみを学習に用いたこと
・判定モデルの精度
・最適画像を用いてinferenceしたときの精度
・最適画像かつsoft voteを用いたときの精度

