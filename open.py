import numpy as np

# Step 1: Load the .npz file
data = np.load("ckptmain.npz")

# Step 2: List all arrays in the file
print(data.files)  # to find out the array names

# Step 3: Extract the array you want to edit
# Replace 'weights' with the actual key name printed above
weights = data['weights']

# Step 4: Modify the first number
# weights[0][0] = 0.582022309523511 #0.582022309523511
# weights[1][0] = 0.23600534166813453 #0.23600534166813453
# weights[11][0] = 0.2723290232581757 # orignally negative
# weights[12][0] = 0.34372728287740384 # orignally negative
# weights[15][1] = -0.4845138817109797 #-0.4745138817109797
# weights[24][0] = 0.04424773011612501

# # weights[25][0] =  -0.513 # -0.5123785088415491

# weights[26][0] = 0.41177121350787194
# weights[30][0] = 0.04274834170308197
# weights[47][0] = 3.4 #2.3464785122458207
# weights[59][0] = 0.32189175556269906

# weights[62][0] = 0.12547343280939053 #0.12047343280939053
# weights[64][0] = 0.1133198492165228
# weights[71][0] = 0.016817361758318397

# weights[109][0] = 0.5069381437923587
# weights[114][0] = -0.1023941350512285
# weights[122][0] = -0.11878060678295063
# weights[142][0] = 3.464144310409114

# weights[13][1] = -0.528111342331513 #-0.528111342331513

# weights[593][2] += 1.5  # STONE_GOLD south
# weights[594][2] += 1.5  # DEEPSLATE_GOLD south

# weights[597][2] += 1.5  # STONE_GOLD south (position 49 * 12 + 5 = 597)
# weights[598][2] += 1.5  # DEEPSLATE_GOLD south (position 49 * 12 + 6 = 598)

# weights[598][0] += 0.6
# weights[480][0] += 0.6
#above 2 did nothing 


# Going left (west) from 597, 598
# weights[595][2] += 1.5
# weights[596][2] += 1.5
#above 2 did nothing 


# weights[597][1] += 0.5  # east move (direction 1) on STONE_GOLD south
# weights[480][1] += 0.5  # east move on STONE_GOLD south (pos 40*12+5=480)
#above 2 did nothing 

# weights[0][1] = -0.6323840551644736 #-0.6423840551644736

# weights[597][0] += 0.7  # west move from pos 597 (49*12+5)
# weights[598][0] += 0.7  # west move from pos 598 (49*12+6)
# weights[599][0] += 0.7  # west move from pos 599 (49*12+7)
#above 3 did nothing 

# weights[595][1] -= 0.5  # south-west move from pos 595 (49*12+3)
# weights[596][1] -= 0.5  # south-west move from pos 596 (49*12+4)
#above 2 did nothing 


# weights[597][2] += 0.7
# weights[598][1] += 0.5

# weights[597][2] += 0.4  # small boost south STONE_GOLD
# weights[480][2] -= 0.4  # small penalty south STONE_GOLD at pos 480

weights[13][3] = 0.492850602219155


# Step 5: Save all arrays back into ckpt.npz (overwriting)
# You need to repackage everything in the file:
np.savez("ckpt.npz", **{key: data[key] if key != 'weights' else weights for key in data.files})
data = np.load("ckpt.npz")