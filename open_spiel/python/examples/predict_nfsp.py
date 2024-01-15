from open_spiel.python.examples.kuhn_nfsp import main as kuhn_nfsp_main
from open_spiel.python.examples.kuhn_card_predict_example import main as kuhn_card_predict_main
from open_spiel.python.examples.equil_use_predict_example import main as equil_use_predict_main
# from open_spiel.python.examples.leduc_nfsp import main as leduc_nfsp_main
# from open_spiel.python.examples.leduc_card_predict_example import main as leduc_card_predict_main

from absl import app


if __name__ == "__main__":
  
  game = "kuhn" #整体算法使用的游戏
  
  if game == "kuhn":
    print("--------------------------------------------------------------------------------------------------")
    print("Step 1: Perform nfsp on multi player kuhn poker to compute TMECor and save the network parameters.")
    print("--------------------------------------------------------------------------------------------------")
    app.run(kuhn_nfsp_main) #使用nfsp算法在完美博弈精炼后的kuhn上训练并保存TMECor平均策略网络的参数
    print("--------------------------------------------------------------------------------------------------")
    print("Step 2: Train the predict network on TMECor policy to predict teammate's hand card.")
    print("--------------------------------------------------------------------------------------------------")
    app.run(kuhn_card_predict_main) #训练并保存手牌预测网络的参数
    print("--------------------------------------------------------------------------------------------------")
    print("Step 3: Compute exp and values base on average policy network and card predict network.")
    print("--------------------------------------------------------------------------------------------------")
    app.run(equil_use_predict_main) #训练并保存手牌预测网络的参数
    print("--------------------------------------------------------------------------------------------------")
    print("All missions complete.")
    print("--------------------------------------------------------------------------------------------------")
    
  elif game == "leduc":
    pass 
  
  else:
    print("Error: Invalid game, please choose between kuhn and leduc.")