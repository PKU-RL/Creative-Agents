from distutils.file_util import move_file
from shlex import join
import minedojo
from minedojo.sim import InventoryItem

#from minedojo.sim.wrappers import fast_reset
import os
import imageio
import numpy as np
import numpy
import math
import utility.FindPath as FP
import pickle

H = 32
W = 32
L = 32
MIN_DIS = 5


S_H = 4
S_W = 881.5
S_L = -924.5

def add_distance(pos_yaw_list, dis):
    if dis<=MIN_DIS:
        dis = int(MIN_DIS)
    # dis -=1
    pos_yaw_list[0][1] -= dis
    pos_yaw_list[0][2] -= dis
    pos_yaw_list[1][1] += 2*dis
    # pos_yaw_list[1][2] -= dis
    # pos_yaw_list[2][1] -= dis
    pos_yaw_list[2][2] += 2*dis
    pos_yaw_list[3][1] -= 2*dis
    # pos_yaw_list[3][2] -= dis
    return pos_yaw_list

def real2grid(x,y,z):
    return [math.floor(y-S_H+0.5),-math.floor(x-S_W+0.5),-math.floor(z-S_L+0.5)]

def grid2real(h,w,l):
    return [-w+S_W,h+S_H,-l+S_L]



forbidden_items = [
    0,
    2,
    3,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    19,
    21,
    23,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
    44,
    46,
    47,
    51,
    52,
    54,
    55,
    56,
    58,
    59,
    60,
    61,
    62,
    63,
    65,
    66,
    68,
    69,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    81,
    83,
    84,
    85,
    86,
    87,
    88,
    92,
    93,
    94,
    96,
    97,
    100,
    104,
    101,
    102,
    104,
    105,
    106,
    107,
    110,
    111,
    113,
    115,
    116,
    117,
    118,
    119,
    120,
    122,
    123,
    124,
    126,
    127,
    129,
    130,
    131,
    132,
    137,
    138,
    139,
    140,
    141,
    142,
    143,
    144,
    145,
    146,
    147,
    148,
    149,
    150,
    151,
    152,
    153,
    154,
    157,
    158,
    166,
    170,
    171,
    175,
    176,
    177,
    178,
    182,
    183,
    184,
    185,
    186,
    187,
    188,
    189,
    190,
    191,
    192,
    193,
    194,
    195,
    196,
    197,
    198,
    207,
    217
]

non_place_items=[
    6,
    12,
    13,
    26,
    27,
    28,
    31,
    32,
    34,
    37,
    38,
    39,
    40,
    50,
    51,
    55,
    59,
    63,
    64,
    65,
    66,
    68,
    69,
    71,
    75,
    76,
    77,
    81,
    83,
    92,
    93,
    94,
    104,
    105,
    106,
    111,
    115,
    116,
    119,
    122,
    132,
    141,
    142,
    143,
    145,
    157,
    166,
    171,
    175,
    178,
    193,
    194,
    195,
    196,
    197,
    198,
    199,
    200,
    205,
    207,
    208,
    209,
    210,
    211,
    213,
    217,
    218,
    235,
    236,
    237,
    238,
    239,
    242,
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    79,
    174,
    212,
    147,
    148,
]

minecraft_items = [
    (0,'air'),
    (1, 'stone'),
    (2, 'grass'),
    (3, 'dirt'),
    (4, 'cobblestone'),
    (5, 'planks'),
    (6, 'sapling'),
    (7, 'bedrock'),
    (8, 'water'),
    (9, 'water'),
    (10, 'lava'),
    (11, 'lava'),
    (12, 'sand'),
    (13, 'gravel'),
    (14, 'gold_ore'),
    (15, 'iron_ore'),
    (16, 'coal_ore'),
    (17, 'log'),
    (18, 'leaves'),
    (19, 'sponge'),
    (20, 'glass'),
    (21, 'lapis_ore'),
    (22, 'lapis_block'),
    (23, 'dispenser'),
    (24, 'sandstone'),
    (25, 'note_block'),
    (26, 'bed'),
    (27, 'powered_rail'),
    (28, 'detector_rail'),
    (29, 'sticky_piston'),
    (30, 'web'),
    (31, 'tall_grass'),
    (32, 'dead_bush'),
    (33, 'piston'),
    (34, 'piston_head'),
    (35, 'wool'),
    (36, 'piston'),
    (37, 'dandelion'),
    (38, 'flower'),
    (39, 'brown_mushroom'),
    (40, 'red_mushroom'),
    (41, 'gold_block'),
    (42, 'iron_block'),
    (43, 'double_stone_slab'),
    (44, 'stone_slab'),
    (45, 'brick_block'),
    (46, 'tnt'),
    (47, 'bookshelf'),
    (48, 'mossy_cobblestone'),
    (49, 'obsidian'),
    (50, 'torch'),
    (51, 'fire'),
    (52, 'mob_spawner'),
    (53, 'oak_stairs'),
    (54, 'chest'),
    (55, 'redstone_wire'),
    (56, 'diamond_ore'),
    (57, 'diamond_block'),
    (58, 'crafting_table'),
    (59, 'wheat'),
    (60, 'farmland'),
    (61, 'furnace'),
    (62, 'lit_furnace'),
    (63, 'standing_sign'),
    (64, 'wooden_door'),
    (65, 'ladder'),
    (66, 'rail'),
    (67, 'stone_stairs'),
    (68, 'wall_sign'),
    (69, 'lever'),
    (70, 'stone_pressure_plate'),
    (71, 'iron_door'),
    (72, 'wooden_pressure_plate'),
    (73, 'redstone_ore'),
    (74, 'lit_redstone_ore'),
    (75, 'unlit_redstone_torch'),
    (76, 'redstone_torch'),
    (77, 'stone_button'),
    (78, 'snow_layer'),
    (79, 'ice'),
    (80, 'snow'),
    (81, 'cactus'),
    (82, 'clay'),
    (83, 'reeds'),
    (84, 'jukebox'),
    (85, 'fence'),
    (86, 'pumpkin'),
    (87, 'netherrack'),
    (88, 'soul_sand'),
    (89, 'glowstone'),
    (90, 'portal'),
    (91, 'lit_pumpkin'),
    (92, 'cake'),
    (93, 'unpowered_repeater'),
    (94, 'powered_repeater'),
    (95, 'stained_glass'),
    (96, 'trapdoor'),
    (97, 'monster_egg'),
    (98, 'stonebrick'),
    (99, 'brown_mushroom_block'),
    (100, 'red_mushroom_block'),
    (101, 'iron_bars'),
    (102, 'glass_pane'),
    (103, 'melon_block'),
    (104, 'pumpkin_stem'),
    (105, 'melon_stem'),
    (106, 'vine'),
    (107, 'fence_gate'),
    (108, 'brick_stairs'),
    (109, 'stone_brick_stairs'),
    (110, 'mycelium'),
    (111, 'waterlily'),
    (112, 'nether_brick'),
    (113, 'nether_brick_fence'),
    (114, 'nether_brick_stairs'),
    (115, 'nether_wart'),
    (116, 'enchanting_table'),
    (117, 'brewing_stand'),
    (118, 'cauldron'),
    (119, 'end_portal'),
    (120, 'end_portal_frame'),
    (121, 'end_stone'),
    (122, 'dragon_egg'),
    (123, 'redstone_lamp'),
    (124, 'lit_redstone_lamp'),
    (125, 'double_wooden_slab'),
    (126, 'wooden_slab'),
    (127, 'cocoa'),
    (128, 'sandstone_stairs'),
    (129, 'emerald_ore'),
    (130, 'ender_chest'),
    (131, 'tripwire_hook'),
    (132, 'tripwire'),
    (133, 'emerald_block'),
    (134, 'spruce_stairs'),
    (135, 'birch_stairs'),
    (136, 'jungle_stairs'),
    (137, 'command_block'),
    (138, 'beacon'),
    (139, 'cobblestone_wall'),
    (140, 'flower_pot'),
    (141, 'carrots'),
    (142, 'potatoes'),
    (143, 'wooden_button'),
    (144, 'skull'),
    (145, 'anvil'),
    (146, 'trapped_chest'),
    (147, 'light_weighted_pressure_plate'),
    (148, 'heavy_weighted_pressure_plate'),
    (149, 'unpowered_comparator'),
    (150, 'powered_comparator'),
    (151, 'daylight_detector'),
    (152, 'redstone_block'),
    (153, 'quartz_ore'),
    (154, 'hopper'),
    (155, 'quartz_block'),
    (156, 'quartz_stairs'),
    (157, 'activator_rail'),
    (158, 'dropper'),
    (159, 'stained_hardened_clay'),
    (160, 'stained_glass_pane'),
    (161, 'leaves2'),
    (162, 'log2'),
    (163, 'acacia_stairs'),
    (164, 'dark_oak_stairs'),
    (165, 'slime'),
    (166, 'barrier'),
    (167, 'iron_trapdoor'),
    (168, 'prismarine'),
    (169, 'sea_lantern'),
    (170, 'hay_block'),
    (171, 'carpet'),
    (172, 'hardened_clay'),
    (173, 'coal_block'),
    (174, 'packed_ice'),
    (175, 'double_plant'),
    (176, 'standing_banner'),
    (177, 'wall_banner'),
    (178, 'daylight_detector_inverted'),
    (179, 'red_sandstone'),
    (180, 'red_sandstone_stairs'),
    (181, 'double_stone_slab2'),
    (182, 'stone_slab2'),
    (183, 'spruce_fence_gate'),
    (184, 'birch_fence_gate'),
    (185, 'jungle_fence_gate'),
    (186, 'dark_oak_fence_gate'),
    (187, 'acacia_fence_gate'),
    (188, 'spruce_fence'),
    (189, 'birch_fence'),
    (190, 'jungle_fence'),
    (191, 'dark_oak_fence'),
    (192, 'acacia_fence'),
    (193, 'spruce_door'),
    (194, 'birch_door'),
    (195, 'jungle_door'),
    (196, 'acacia_door'),
    (197, 'dark_oak_door'),
    (198, 'end_rod'),
    (199, 'chorus_plant'),
    (200, 'chorus_flower'),
    (201, 'purpur_block'),
    (202, 'purpur_pillar'),
    (203, 'purpur_stairs'),
    (204, 'purpur_double_slab'),
    (205, 'purpur_slab'),
    (206, 'end_bricks'),
    (207, 'beetroots'),
    (208, 'grass_path'),
    (209, 'end_gateway'),
    (210, 'repeating_command_block'),
    (211, 'chain_command_block'),
    (212, 'frosted_ice'),
    (213, 'magma'),
    (214, 'nether_wart_block'),
    (215, 'red_nether_brick'),
    (216, 'bone_block'),
    (217, 'structure_void'),
    (218, 'observer'),
    (219, 'white_shulker_box'),
    (220, 'orange_shulker_box'),
    (221, 'magenta_shulker_box'),
    (222, 'light_blue_shulker_box'),
    (223, 'yellow_shulker_box'),
    (224, 'lime_shulker_box'),
    (225, 'pink_shulker_box'),
    (226, 'gray_shulker_box'),
    (227, 'light_gray_shulker_box'),
    (228, 'cyan_shulker_box'),
    (229, 'purple_shulker_box'),
    (230, 'blue_shulker_box'),
    (231, 'brown_shulker_box'),
    (232, 'green_shulker_box'),
    (233, 'red_shulker_box'),
    (234, 'black_shulker_box'),
    (235, 'white_glazed_terracotta'),
    (236, 'orange_glazed_terracotta'),
    (237, 'magenta_glazed_terracotta'),
    (238, 'light_blue_glazed_terracotta'),
    (239, 'yellow_glazed_terracotta'),
    (240, 'lime_glazed_terracotta'),
    (241, 'pink_glazed_terracotta'),
    (242, 'gray_glazed_terracotta'),
    (243, 'light_gray_glazed_terracotta'),
    (244, 'cyan_glazed_terracotta'),
    (245, 'purple_glazed_terracotta'),
    (246, 'blue_glazed_terracotta'),
    (247, 'brown_glazed_terracotta'),
    (248, 'green_glazed_terracotta'),
    (249, 'red_glazed_terracotta'),
    (250, 'black_glazed_terracotta'),
    (251, 'concrete'),
    (252, 'concrete_powder'),
    (253, 'structure_block'),
    (255, 'iron_shovel'),
    (256, 'iron_pickaxe'),
    (257, 'iron_axe'),
    (258, 'flint_and_steel'),
    (259, 'apple'),
    (260, 'bow'),
    (261, 'arrow'),
    (262, 'coal'),
    (263, 'diamond'),
    (264, 'iron_ingot'),
    (265, 'gold_ingot'),
    (266, 'iron_sword'),
    (267, 'wooden_sword'),
    (268, 'wooden_shovel'),
    (269, 'wooden_pickaxe'),
    (270, 'wooden_axe'),
    (271, 'stone_sword'),
    (272, 'stone_shovel'),
    (273, 'stone_pickaxe'),
    (274, 'stone_axe'),
    (275, 'diamond_sword'),
    (276, 'diamond_shovel'),
    (277, 'diamond_pickaxe'),
    (278, 'diamond_axe'),
    (279, 'stick'),
    (280, 'bowl'),
    (281, 'mushroom_stew'),
    (282, 'golden_sword'),
    (283, 'golden_shovel'),
    (284, 'golden_pickaxe'),
    (285, 'golden_axe'),
    (286, 'string'),
    (287, 'feather'),
    (288, 'gunpowder'),
    (289, 'wooden_hoe'),
    (290, 'stone_hoe'),
    (291, 'iron_hoe'),
    (292, 'diamond_hoe'),
    (293, 'golden_hoe'),
    (294, 'wheat_seeds'),
    (295, 'wheat'),
    (296, 'bread'),
    (297, 'leather_helmet'),
    (298, 'leather_chestplate'),
    (299, 'leather_leggings'),
    (300, 'leather_boots'),
    (301, 'chainmail_helmet'),
    (302, 'chainmail_chestplate'),
    (303, 'chainmail_leggings'),
    (304, 'chainmail_boots'),
    (305, 'iron_helmet'),
    (306, 'iron_chestplate'),
    (307, 'iron_leggings'),
    (308, 'iron_boots'),
    (309, 'diamond_helmet'),
    (310, 'diamond_chestplate'),
    (311, 'diamond_leggings'),
    (312, 'diamond_boots'),
    (313, 'golden_helmet'),
    (314, 'golden_chestplate'),
    (315, 'golden_leggings'),
    (316, 'golden_boots'),
    (317, 'flint'),
    (318, 'porkchop'),
    (319, 'cooked_porkchop'),
    (320, 'painting'),
    (321, 'golden_apple'),
    (322, 'sign'),
    (323, 'wooden_door'),
    (324, 'bucket'),
    (325, 'water_bucket'),
    (326, 'lava_bucket'),
    (327, 'minecart'),
    (328, 'saddle'),
    (329, 'iron_door'),
    (330, 'redstone'),
    (331, 'snowball'),
    (332, 'boat'),
    (333, 'leather'),
    (334, 'milk_bucket'),
    (335, 'brick'),
    (336, 'clay_ball'),
    (337, 'reeds'),
    (338, 'paper'),
    (339, 'book'),
    (340, 'slime_ball'),
    (341, 'chest_minecart'),
    (342, 'furnace_minecart'),
    (343, 'egg'),
    (344, 'compass'),
    (345, 'fishing_rod'),
    (346, 'clock'),
    (347, 'glowstone_dust'),
    (348, 'fish'),
    (349, 'cooked_fish'),
    (350, 'dye'),
    (351, 'bone'),
    (352, 'sugar'),
    (353, 'cake'),
    (354, 'bed'),
    (355, 'repeater'),
    (356, 'cookie'),
    (357, 'filled_map'),
    (358, 'shears'),
    (359, 'melon'),
    (360, 'pumpkin_seeds'),
    (361, 'melon_seeds'),
    (362, 'beef'),
    (363, 'cooked_beef'),
    (364, 'chicken'),
    (365, 'cooked_chicken'),
    (366, 'rotten_flesh'),
    (367, 'ender_pearl'),
    (368, 'blaze_rod'),
    (369, 'ghast_tear'),
    (370, 'gold_nugget'),
    (371, 'nether_wart'),
    (372, 'potion'),
    (373, 'glass_bottle'),
    (374, 'spider_eye'),
    (375, 'fermented_spider_eye'),
    (376, 'blaze_powder'),
    (377, 'magma_cream'),
    (378, 'brewing_stand'),
    (379, 'cauldron'),
    (380, 'ender_eye'),
    (381, 'speckled_melon'),
    (382, 'spawn_egg'),
    (383, 'experience_bottle'),
    (384, 'fire_charge'),
    (385, 'writable_book'),
    (386, 'written_book'),
    (387, 'emerald'),
    (388, 'item_frame'),
    (389, 'flower_pot'),
    (390, 'carrot'),
    (391, 'potato'),
    (392, 'baked_potato'),
    (393, 'poisonous_potato'),
    (394, 'map'),
    (395, 'golden_carrot'),
    (396, 'skull'),
    (397, 'carrot_on_a_stick'),
    (398, 'nether_star'),
    (399, 'pumpkin_pie'),
    (400, 'fireworks'),
    (401, 'firework_charge'),
    (402, 'enchanted_book'),
    (403, 'comparator'),
    (404, 'netherbrick'),
    (405, 'quartz'),
    (406, 'tnt_minecart'),
    (407, 'hopper_minecart'),
    (408, 'prismarine_shard'),
    (409, 'prismarine_crystals'),
    (410, 'rabbit'),
    (411, 'cooked_rabbit'),
    (412, 'rabbit_stew'),
    (413, 'rabbit_foot'),
    (414, 'rabbit_hide'),
    (415, 'armor_stand'),
    (416, 'iron_horse_armor'),
    (417, 'golden_horse_armor'),
    (418, 'diamond_horse')
]

def simplify_type(type):
    if type not in forbidden_items and type not in non_place_items:
        return type
    else:
        return 1

def jump_and_place_down(env, img_list, block_type = 1):
    block_type = simplify_type(block_type) # ALL_ITEM bug in minedojo, use stone instead
    # equip = '/give @p {}'.format(minecraft_items[block_type][1])
    # env.execute_cmd(equip)
    initial_inv = [
        InventoryItem(slot=1, name=minecraft_items[int(block_type)][1], variant=None, quantity=1),
    ]
    env.set_inventory(initial_inv)
    action_list_place = []

    HOLD_STEPS = 2
    LOOKDOWN_STEPS = 3

    #look down
    for j in range(LOOKDOWN_STEPS):
        action = env.action_space.no_op()
        action[3]=23
        action_list_place.append(action)
    # #break grass
    # for j in range(5):
    #     action = env.action_space.no_op()
    #     action[5]=3
    #     action_list_place.append(action)

    #jump and place
    for j in range(2):
        action = env.action_space.no_op()
        action[2]=1
        action[5]=6
        action[7]=1
        action_list_place.append(action)

    for j in range(HOLD_STEPS):
        action = env.action_space.no_op()
        action_list_place.append(action)

    my_len = len(action_list_place)
    for j in range(my_len):
        action = action_list_place[j]
        obs, reward, done, info = env.step(action)
        # print(obs["location_stats"]["pos"])
        img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
    
    return img_list

def fix_pos(env):
    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    x,y,z = obs["location_stats"]["pos"]
    tempx,tempy,tempz = real2grid(x,y,z)
    nx,ny,nz = grid2real(tempx,tempy,tempz)
    # print(nz)
    tp = f'/tp @p {nx} {ny} {nz}'
    env.execute_cmd(tp)
    HOLD_STEPS = 3
    for j in range(HOLD_STEPS):
        action = env.action_space.no_op()
        obs, reward, done, info = env.step(action)
    
    return

def direction2yaw(env,direction):
    goal_yaw = 0
    if direction in [(0,0,1),(1,0,1),(-1,0,1)]:
        goal_yaw =  0
    elif(direction in [(0,0,-1),(1,0,-1),(-1,0,-1)]):
        goal_yaw =  12
    elif(direction in [(0,1,0),(1,1,0),(-1,1,0)]):
        goal_yaw =  6
    elif(direction in [(0,-1,0),(1,-1,0),(-1,-1,0)]):
        goal_yaw =  18
    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    yaw = int((-obs["location_stats"]["yaw"][0] )/15)
    # env.reset()

    action = env.action_space.no_op()
    # print(goal_yaw, yaw)
    if yaw<0:
        yaw+=24
    if goal_yaw<yaw:
        goal_yaw+=24
    # action[4] = 0
    action[4] = 24-(goal_yaw - yaw)
    # print(action[4])
    # print(action[4])
    obs, reward, done, info = env.step(action)

    return

def move(env,img_list,pos=[0,0,0]):
    action_list_place = []
    HOLD_STEPS = 4
    #move forward
    for j in range(HOLD_STEPS):
        action = env.action_space.no_op()
        action[0]=1
        action_list_place.append(action)

    for j in range(HOLD_STEPS):
        action = env.action_space.no_op()
        action_list_place.append(action)

    my_len = len(action_list_place)
    for j in range(my_len):
        action = action_list_place[j]
        obs, reward, done, info = env.step(action)
        # print(obs["location_stats"]["pos"])
        img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
        # print(obs["voxels"]["block_name"].shape)
    
    return img_list

def move_jump(env,img_list,pos=[0,0,0]):
    action_list_place = []
    HOLD_STEPS = 4

    #move forward
    for j in range(2):
        action = env.action_space.no_op()
        action[0]=1
        action[2]=1
        action_list_place.append(action)
    for j in range(1):
        action = env.action_space.no_op()
        action[0]=1
        action[2]=1
        action_list_place.append(action)
    for j in range(2):
        action = env.action_space.no_op()
        action[0]=1
        action[2]=1
        action_list_place.append(action)

    for j in range(HOLD_STEPS):
        action = env.action_space.no_op()
        action_list_place.append(action)

    my_len = len(action_list_place)
    for j in range(my_len):
        action = action_list_place[j]
        obs, reward, done, info = env.step(action)
        # print(obs["location_stats"]["pos"])
        # print(obs["location_stats"]["yaw"])
        img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
        # print(obs["voxels"]["block_name"].shape)
    
    return img_list

def get_pos(env):
    action = env.action_space.no_op()
    obs, reward, done, info = env.step(action)
    return(obs["location_stats"]["pos"])

def place_block(env,h,w,l,type):
    type = simplify_type(type) # ALL_ITEM bug in minedojo, use stone instead
    xyz = grid2real(h,w,l)
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    x-=0.5
    z-=0.5
    support = f'/setblock {int(x)} {int(y)} {int(z)} minecraft:{minecraft_items[int(type)][1]}'
    env.execute_cmd(support)

def update_path(vox_built):
    grid_path = np.zeros((H,W,L))
    for j in range(W):
        for k in range(L):
            for i in range(H):
                if vox_built[i,j,k]==0 and (i+1>=H or vox_built[i+1,j,k]==0) and (i-1<0 or vox_built[i-1,j,k]==1):
                    grid_path[i,j,k]=1
    return grid_path

def ground_data(data):
    min_h = H
    for i in data:
        if i[2]<min_h:
            min_h = i[2]
    for i in data:
        i[2] = i[2]-min_h
    return data


def Build(seq_path="./results/seq",save_pth = "./results/building",full_record = True, append = True):
    seq_files = os.listdir(seq_path)
    pkl_files = [file for file in seq_files if file.endswith(".pkl")]
    
    seed=6+1
    # save_pth="./"
    np.random.seed(seed)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)    
    

    for i in pkl_files:
        env = minedojo.make(
            task_id="open-ended",
            image_size=(288, 512),
            seed=seed,
            # use_voxel = True,
            # voxel_size = dict(xmin = -33,ymin = -33, zmin = -33, xmax = 33, ymax = 33, zmax = 33),
            fast_reset = False,
            # fast_reset=True,# set teleport_reset mode
            generate_world_type = "flat",
            allow_time_passage = False,
            allow_mob_spawn = False,
        )

        env.reset()
        spectator = f'/gamemode spectator'
        obs, reward, done, info = env.execute_cmd(spectator)
        img_list = []

        # for l in range(4):
        action = env.action_space.no_op()
        action[4] = 0
        obs, reward, done, info = env.step(action)
        # print(obs["location_stats"]["yaw"])


        with open(os.path.join(seq_path,i), 'rb') as f:
            data = pickle.load(f)
            print("Total voxel:{}".format(len(data)))
            data = ground_data(data)
            vox_built = np.zeros((H,W,L))
            grid_path = np.concatenate((np.ones((1,W,L)),np.zeros((H-1,W,L))),axis=0)

            empty_flag = True

            for k in data:
                temp = get_pos(env)
                pos = real2grid(temp[0],temp[1],temp[2])
                if vox_built[k[2],k[1],k[0]] == 0 :
                    pth =  FP.find_path(pos,[k[2],k[1],k[0]],grid_path)
                    if pth is not None:
                        empty_flag = False
                        # place_block(env,k[2],k[1],k[0],k[3])
                        # print(pth)
                        for i_ in range(len(pth)-1):
                            direction = (pth[i_+1][0]-pth[i_][0],pth[i_+1][1]-pth[i_][1],pth[i_+1][2]-pth[i_][2])
                            direction2yaw(env,direction)
                            if direction[0]>0:
                                img_list = move_jump(env,img_list)
                            else:
                                img_list = move(env,img_list)
                            fix_pos(env)
                        temp = get_pos(env)
                        pos = real2grid(temp[0],temp[1],temp[2])
                        # print(pos,[k[2],k[1],k[0]])
                        if pos == [k[2],k[1],k[0]]:
                            img_list = jump_and_place_down(env,img_list,k[3])
                        else:
                            place_block(env,k[2],k[1],k[0],k[3])
                        vox_built[k[2],k[1],k[0]] = 1
                        grid_path = update_path(vox_built)
                if append:
                    place_block(env,k[2],k[1],k[0],k[3])
            if empty_flag:
                print("EMPTY VOXEL TO BUILD, SKIP")
                env.reset()
                env.close()
                continue

            tp = f'/tp @p {int(S_W)} {int(S_H)} {int(S_L)}'
            env.execute_cmd(tp)

            real_width = 25
            view_angle = 35
            real_h = 20
            fit_dis = real_width / math.tan(math.pi * (view_angle/180))
            fit_pitch = math.atan(real_h*3/4/fit_dis)/math.pi * 180
            fit_pitch_num = math.ceil(fit_pitch / 15)
            # print("fit_pitch:{}".format(fit_pitch))
            # print("fit_pitch_num:{}".format(fit_pitch_num))

            fit_dis = fit_dis / math.sqrt(2)
            pitch_list = [12,12,12,12]
            pitch_list[0]+=fit_pitch_num
            # pitch_list = [[14,12,12,12],[15,12,12,12],[16,12,12,12],[17,12,12,12]] #-60, -45, -30, -15
            

            # print("real_h:{}".format(real_h))
            POS_YAW_LIST = [[int(real_h),0,0,9],[0,W,0,18],[0,0,L,18],[0,-W,0,18]] # y,x,z,yaw
            pos_yaw_list = add_distance(POS_YAW_LIST, int(fit_dis-(W+L)/4))

            support = f'/setblock ~ ~{-2} ~ minecraft:glass'
            remove_support = f'/fill ~ ~{-1} ~ ~ ~{-2} ~ minecraft:air'

            segment = 1

            warm_wait = 30
            warm_wait_flag = True
            gif_list = []
            WAIT_TIME  = 5
            steps = 4

            tp_pos = [grid2real(25,0,0),grid2real(25,0,-15),grid2real(25,-15,-15),grid2real(25,-15,0)]

            for k in range(steps*segment):
                action = env.action_space.no_op()

                # Action Space : https://docs.minedojo.org/sections/core_api/action_space.html
                done = False

                if k%segment == 0 :
                    k = int(k / segment)

                    # tp = f'/tp @p ~{pos_yaw_list[k][1]} ~{pos_yaw_list[k][0]} ~{pos_yaw_list[k][2]} '
                    tp = f'/tp @p {tp_pos[k][0]} {tp_pos[k][1]} {tp_pos[k][2]}'

                    env.execute_cmd(tp)
                    env.execute_cmd(support)

                    goal_yaw = [3,21,15,9]

                    action = env.action_space.no_op()
                    obs, reward, done, info = env.step(action)
                    yaw = int((-obs["location_stats"]["yaw"][0] )/15)
                    # env.reset()

                    action = env.action_space.no_op()
                    # print(goal_yaw, yaw)
                    if yaw<0:
                        yaw+=24
                    if goal_yaw[k]<yaw:
                        goal_yaw[k]+=24
                    # action[4] = 0
                    action[4] = 24-(goal_yaw[k] - yaw)


                    # # action[3] = pitch_list[k]
                    if k==0:
                        action[3] = 15
                    # action[4] = 24 - pos_yaw_list[k][3]
                    
                    obs, reward, done, info = env.step(action)

                    temp_img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                    gif_list.append(temp_img)
                    # print(os.path.splitext(i))
                    imageio.mimsave(os.path.join(save_pth,str(os.path.splitext(i)[0])+'_view.gif'), gif_list, duration=0.7)
                        
        if full_record:      
            pth = os.path.join(save_pth, '{}_full.gif'.format(i))
            imageio.mimsave(pth, img_list, duration=0.15)
        env.reset()
        env.close()
        print("[INFO] Test Success")


if __name__ == "__main__":
    Build()