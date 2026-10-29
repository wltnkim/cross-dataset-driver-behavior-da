# Confusion matrices (fold 0) transcribed from the per-run confusion_matrix*.png in results/.
# Rows = true, cols = predicted, class order: drinking, phone_call, phone_texting, reaching, adjusting_controls, seatbelt.
# 5x5 matrices omit seatbelt (absent from DMD/3MDAD ground truth and never predicted by these models' plots).
R="results/"
SO="20260604_235447_cross_dann_r000_srconlysv_fold0/"
CM={
 ("driveandact","Source-Only"):(SO+"classification_report_eval_driveandact_fold0.txt",
   [[0,16,22,26,10,0],[4,34,11,20,5,11],[0,49,14,14,1,5],[5,1,8,233,19,2],[42,17,42,13,20,9],[2,0,0,28,6,0]]),
 ("driveandact","DANN"):("20260324_144811_cross_dann_fold0/classification_report.txt",
   [[71,0,3,0,0,0],[0,76,1,1,7,0],[0,1,80,2,0,0],[0,1,5,259,1,2],[0,0,0,0,143,0],[0,0,0,3,0,33]]),
 ("driveandact","Target-Only"):("20260323_183659_cross_target_only_fold0/classification_report.txt",
   [[72,0,0,2,0,0],[0,74,1,5,4,1],[0,2,77,4,0,0],[1,0,5,261,0,1],[0,0,0,1,142,0],[0,0,0,3,0,33]]),
 ("driveandact","CleanAdapt"):("20260326_185549_cross_cleanadapt_fold0/classification_report.txt",
   [[73,0,1,0,0,0],[1,69,1,0,14,0],[0,4,79,0,0,0],[10,7,6,235,4,6],[0,1,0,0,142,0],[0,0,0,1,1,34]]),
 ("dmd","Source-Only"):(SO+"classification_report_eval_dmd_fold0.txt",
   [[3,1,116,123,20,0],[79,273,381,628,293,45],[54,268,154,135,151,5],[36,30,42,851,206,3],[57,19,42,18,54,9],[0,0,0,0,0,0]]),
 ("dmd","DANN"):("20260330_140718_cross_dann_dmd_fold0/classification_report.txt",
   [[262,0,1,0,0],[0,1677,10,11,1],[0,11,747,8,1],[14,5,32,1117,0],[0,5,0,3,191]]),
 ("dmd","Target-Only"):("20260330_135417_cross_target_only_dmd_fold0/classification_report.txt",
   [[253,7,3,0,0],[1,1667,29,0,2],[0,10,751,5,1],[12,7,36,1108,5],[0,4,0,0,195]]),
 ("dmd","CleanAdapt"):("20260405_213125_cross_cleanadapt_dmd_fold0/classification_report.txt",
   [[263,0,0,0,0],[0,1667,12,19,1],[1,6,752,7,1],[18,4,27,1119,0],[0,3,4,1,191]]),
 ("3mdad","DANN"):("20260402_091133_cross_dann_3mdad_fold0/classification_report.txt",
   [[32,0,0,0,0],[0,29,0,1,2],[0,0,32,0,0],[0,0,0,16,0],[1,0,0,0,31]]),
 ("3mdad","Target-Only"):("20260402_091133_cross_target_only_3mdad_fold0/classification_report.txt",
   [[29,0,3,0,0],[0,29,2,0,1],[0,0,32,0,0],[0,0,1,15,0],[0,0,3,0,29]]),
 ("3mdad","CleanAdapt"):("20260405_193433_cross_cleanadapt_3mdad_fold0/classification_report.txt",
   [[32,0,0,0,0],[1,31,0,0,0],[0,0,32,0,0],[0,0,0,16,0],[2,0,0,0,30]]),
}
