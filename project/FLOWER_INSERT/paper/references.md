# References（依正文出現順序編號，已核對 DBLP/官方 BibTeX 修正格式）

> IEEE 格式。編號依正文首次出現順序排列：I.B → I.C → II.D → III.C → IV.A → V.
> [10]、[11] 為目前正文尚未實際引用之補充文獻，暫列於最後；若定稿前仍未引用，建議直接移除。

[1] D. Amodei, C. Olah, J. Steinhardt, P. F. Christiano, J. Schulman, and D. Mané,
"Concrete problems in AI safety," arXiv preprint arXiv:1606.06565, 2016.
— 用途：I.B / III.B，reward hacking 一詞的原始出處。
— 已核對 DBLP BibTeX：作者為 "Paul F. Christiano"（含中間名縮寫 F.）。

[2] J. Skalse, N. H. R. Howe, D. Krasheninnikov, and D. Krueger,
"Defining and characterizing reward gaming," in Proc. 36th Conf. Neural
Inf. Process. Syst. (NeurIPS), New Orleans, LA, pp. 9460–9471, 2022.
— 用途：I.B / III.B，reward hacking／reward gaming 較新、較形式化的數學定義。
— 已核對官方引用：正確標題為 "reward gaming"（非 "reward hacking"）；
已查證 NeurIPS 2022 舉辦地點（New Orleans），改為研討會格式並補上地點與頁碼。

[3] S. M. Devlin and D. Kudenko, "Dynamic potential-based reward shaping,"
in Proc. 11th Int. Conf. Autonomous Agents and Multiagent Systems (AAMAS),
Valencia, Spain, pp. 433–440, 2012.
— 用途：I.C 研究方法概述，Multi-Stage Reward Shaping 框架提出處，
可用於說明本研究之階段式獎勵設計與經典 reward shaping 理論的關聯。
— 已補上會議地點（Valencia, Spain，經查證 AAMAS 2012 確實在此舉辦）。

[4] Y. Bengio, J. Louradour, R. Collobert, and J. Weston, "Curriculum learning,"
in Proc. 26th Annu. Int. Conf. Machine Learning (ICML), Montreal, QC, Canada,
pp. 41–48, 2009.
— 用途：I.C 研究方法概述，「隱式課程學習（implicit curriculum learning）」一詞之
理論出處，說明本研究事件驅動狀態機與經典課程學習理論之關聯。
— 已依 ACM BibTeX 補上會議地點（Montreal, Quebec, Canada）。

[5] R. Toro Icarte, T. Q. Klassen, R. Valenzano, and S. A. McIlraith,
"Reward machines: Exploiting reward function structure in reinforcement learning,"
J. Artif. Intell. Res., vol. 73, pp. 173–208, 2022.
— 用途：II.D Event-Driven State Machine Design，「事件驅動有限狀態機」設計之理論出處。

[6] R. S. Sutton and A. G. Barto, Reinforcement Learning: An Introduction, 2nd ed.
Cambridge, MA: MIT Press, 2018.
— 用途：III.C Local Optimum Trap，探索-利用權衡（exploration-exploitation trade-off）
之經典教科書出處。

[7] M. Mittal et al.,
"Orbit: A unified simulation framework for interactive robot learning environments,"
IEEE Robot. Autom. Lett., vol. 8, no. 6, pp. 3740–3747, 2023.
— 用途：IV.A Training Configuration，Isaac Lab（前身 Orbit）模擬平台之框架論文引用依據。

[8] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov,
"Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, 2017.
— 用途：IV.A Training Configuration，PPO 演算法原始論文。

[9] A. Y. Ng, D. Harada, and S. Russell, "Policy invariance under reward
transformations: Theory and application to reward shaping," in Proc. 16th Int.
Conf. Machine Learning (ICML), Bled, Slovenia, pp. 278–287, 1999.
— 用途：V. Conclusion Limitation 段落，potential-based reward shaping 與
policy invariance 理論之原始出處，支撐「本研究未嚴格遵循此形式化定義」之敘述。
— 已補上會議地點（Bled, Slovenia，經查證 ICML 1999 確實在此舉辦）。

[10] J. Clark and D. Amodei, "Faulty reward functions in the wild," OpenAI Blog, Dec. 2016.
— 用途：正文目前未實際引用；可選用於 I.B／III.B 補充 reward hacking 具體案例
（CoastRunners 賽船遊戲）。（未引用，暫列最後）

[11] V. Krakovna et al., "Specification gaming: The flip side of AI ingenuity,"
DeepMind Blog, Apr. 2020.
— 用途：正文目前未實際引用；reward hacking 的近義詞「specification gaming」，
可選用於概念對照。（未引用，暫列最後）

---

## 尚待處理

- [ ] IV.A 需補上 [7]（Mittal, Isaac Lab）在 "Isaac Lab simulation platform" 處，
  並將原本掛在 PPO 上的引用改為 [8]
- [ ] III.C 需補上 [6]（Sutton & Barto）在 exploration-exploitation trade-off 處
- [ ] I.C 需補上 [4]（Bengio）在 "implicit curriculum learning" 處
- [ ] V. Conclusion 需補上 [9]（Ng, Harada, Russell）在 Limitation 段落
- [ ] 稀疏獎勵問題（sparse reward problem）一般性定義來源（仍未補）
- [ ] 機器人操作＋強化學習相關研究（grasping/stacking/assembly），需自行挑選 2~3 篇貼合領域之代表文獻
- [~] RND（Random Network Distillation）— 已決定不在論文正文提及
- [ ] [10]、[11] 若最終未在正文引用，投稿前應考慮直接移除，避免出現未引用之參考文獻
