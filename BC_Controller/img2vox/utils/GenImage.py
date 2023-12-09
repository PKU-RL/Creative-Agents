from distutils.file_util import move_file
from shlex import join
import minedojo
#from minedojo.sim.wrappers import fast_reset
import os
import imageio
import numpy as np
import numpy
import math
import random

import numpy as np 
import cv2
import pickle
from scipy.ndimage import zoom
import os
import skimage.measure as measure

H = 32
W = 32
L = 32


MIN_SEG_SIZE = 20
DXYZ = [[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
DXYZ = np.array(DXYZ)



def check_bound(xyz,input_shape):
    for i in range(3):
        # print(xyz[i])
        # print(input_shape)
        if xyz[i]<0 or xyz[i]>=input_shape[i]:
            return False
    return True


non_place_items=[
    6,
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
    119,
    122,
    132,
    141,
    142,
    143,
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
    217,

]




non_place_items=[
    6,
    8,
    9,
    10,
    11,
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
    119,
    122,
    132,
    141,
    142,
    143,
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
    217,
]


def remove_noise(voxel_data, output = False):

    # return voxel_data

    for k in non_place_items:
        non_place_ones = np.where(voxel_data == k)
        if non_place_ones[0].size!=0:
            for j in range(non_place_ones[0].size):
                j = (non_place_ones[0][j],non_place_ones[1][j],non_place_ones[2][j])
                flag = False
                for i in range(4):
                    # print(DXYZ[i])
                    j += DXYZ[i]
                    j = tuple(j)
                    # print(j)
                    if check_bound(j,voxel_data.shape):
                        if voxel_data[j]>0 and voxel_data[j] not in non_place_items:
                            flag = True
                            break
                if not flag:
                    voxel_data[j]=0

    ones = np.where(voxel_data>0.5,1,0)
    labeled = measure.label(ones,connectivity=1)
    total_seg = np.max(labeled)

    # print(total_seg)
    for i in range(1,total_seg+1):
        ordered_seg = np.where(labeled==i,1,0)
        temp_sum = np.sum(ordered_seg)
        if temp_sum < MIN_SEG_SIZE:
            labeled -= ordered_seg*i

    if output:
        for j in range(voxel_data.shape[0]):
            np.savetxt('combined_{}.txt'.format(j), labeled[j], fmt = '%d')
    # new_ones = np.where(labeled>0,1,0)
    # if(new_ones.sum()<ones.sum()):
    #     print("{} --> {}".format(ones.sum(),new_ones.sum()))

    voxel_data = np.where(labeled>0,voxel_data,0)
    # print(total_seg)
    # print(label(voxel_data))
    return voxel_data


DPOS = [-3,-2,-1,0,1,2,3]
D_P = [0.08,0.14,0.17,0.22,0.17,0.14,0.08]
def random_xyz():
    xyz = []
    for i in range(3):
        xyz.append(np.random.choice(np.arange(-3,4),p = D_P))
    return xyz

# for i in range(100):
#     print(random_xyz())

WEATHER_CLEAR = True

forbidden_items = [
    6,
    8,
    9,
    10,
    11,
    26,
    28,
    31,
    32,
    34,
    37,
    38,
    39,
    40,
    46,
    51,
    55,
    77,
    83,
    93,
    94,
    104,
    105,
    106,
    111,
    115,
    119,
    122,
    131,
    132,
    141,
    142,
    143,
    149,
    150,
    154,
    157,
    166,
    175,
    176,
    177,
    198,
    207,
    217
]

minecraft_items = [
    (0,'minecraft:air'),
    (1, 'minecraft:stone'),
    (2, 'minecraft:grass_block'),
    (3, 'minecraft:dirt'),
    (4, 'minecraft:cobblestone'),
    (5, 'minecraft:planks'),
    (6, 'minecraft:sapling'),
    (7, 'minecraft:bedrock'),
    (8, 'minecraft:water'),
    (9, 'minecraft:water'),
    (10, 'minecraft:lava'),
    (11, 'minecraft:lava'),
    (12, 'minecraft:sand'),
    (13, 'minecraft:gravel'),
    (14, 'minecraft:gold_ore'),
    (15, 'minecraft:iron_ore'),
    (16, 'minecraft:coal_ore'),
    (17, 'minecraft:log'),
    (18, 'minecraft:leaves'),
    (19, 'minecraft:sponge'),
    (20, 'minecraft:glass'),
    (21, 'minecraft:lapis_ore'),
    (22, 'minecraft:lapis_block'),
    (23, 'minecraft:dispenser'),
    (24, 'minecraft:sandstone'),
    (25, 'minecraft:note_block'),
    (26, 'minecraft:bed'),
    (27, 'minecraft:powered_rail'),
    (28, 'minecraft:detector_rail'),
    (29, 'minecraft:sticky_piston'),
    (30, 'minecraft:web'),
    (31, 'minecraft:tall_grass'),
    (32, 'minecraft:dead_bush'),
    (33, 'minecraft:piston'),
    (34, 'minecraft:piston_head'),
    (35, 'minecraft:wool'),
    (36, 'minecraft:piston'),
    (37, 'minecraft:dandelion'),
    (38, 'minecraft:flower'),
    (39, 'minecraft:brown_mushroom'),
    (40, 'minecraft:red_mushroom'),
    (41, 'minecraft:gold_block'),
    (42, 'minecraft:iron_block'),
    (43, 'minecraft:double_stone_slab'),
    (44, 'minecraft:stone_slab'),
    (45, 'minecraft:brick_block'),
    (46, 'minecraft:tnt'),
    (47, 'minecraft:bookshelf'),
    (48, 'minecraft:mossy_cobblestone'),
    (49, 'minecraft:obsidian'),
    (50, 'minecraft:torch'),
    (51, 'minecraft:fire'),
    (52, 'minecraft:mob_spawner'),
    (53, 'minecraft:oak_stairs'),
    (54, 'minecraft:chest'),
    (55, 'minecraft:redstone_wire'),
    (56, 'minecraft:diamond_ore'),
    (57, 'minecraft:diamond_block'),
    (58, 'minecraft:crafting_table'),
    (59, 'minecraft:wheat'),
    (60, 'minecraft:farmland'),
    (61, 'minecraft:furnace'),
    (62, 'minecraft:lit_furnace'),
    (63, 'minecraft:standing_sign'),
    (64, 'minecraft:wooden_door'),
    (65, 'minecraft:ladder'),
    (66, 'minecraft:rail'),
    (67, 'minecraft:stone_stairs'),
    (68, 'minecraft:wall_sign'),
    (69, 'minecraft:lever'),
    (70, 'minecraft:stone_pressure_plate'),
    (71, 'minecraft:iron_door'),
    (72, 'minecraft:wooden_pressure_plate'),
    (73, 'minecraft:redstone_ore'),
    (74, 'minecraft:lit_redstone_ore'),
    (75, 'minecraft:unlit_redstone_torch'),
    (76, 'minecraft:redstone_torch'),
    (77, 'minecraft:stone_button'),
    (78, 'minecraft:snow_layer'),
    (79, 'minecraft:ice'),
    (80, 'minecraft:snow'),
    (81, 'minecraft:cactus'),
    (82, 'minecraft:clay'),
    (83, 'minecraft:reeds'),
    (84, 'minecraft:jukebox'),
    (85, 'minecraft:fence'),
    (86, 'minecraft:pumpkin'),
    (87, 'minecraft:netherrack'),
    (88, 'minecraft:soul_sand'),
    (89, 'minecraft:glowstone'),
    (90, 'minecraft:portal'),
    (91, 'minecraft:lit_pumpkin'),
    (92, 'minecraft:cake'),
    (93, 'minecraft:unpowered_repeater'),
    (94, 'minecraft:powered_repeater'),
    (95, 'minecraft:stained_glass'),
    (96, 'minecraft:trapdoor'),
    (97, 'minecraft:monster_egg'),
    (98, 'minecraft:stonebrick'),
    (99, 'minecraft:brown_mushroom_block'),
    (100, 'minecraft:red_mushroom_block'),
    (101, 'minecraft:iron_bars'),
    (102, 'minecraft:glass_pane'),
    (103, 'minecraft:melon_block'),
    (104, 'minecraft:pumpkin_stem'),
    (105, 'minecraft:melon_stem'),
    (106, 'minecraft:vine'),
    (107, 'minecraft:fence_gate'),
    (108, 'minecraft:brick_stairs'),
    (109, 'minecraft:stone_brick_stairs'),
    (110, 'minecraft:mycelium'),
    (111, 'minecraft:waterlily'),
    (112, 'minecraft:nether_brick'),
    (113, 'minecraft:nether_brick_fence'),
    (114, 'minecraft:nether_brick_stairs'),
    (115, 'minecraft:nether_wart'),
    (116, 'minecraft:enchanting_table'),
    (117, 'minecraft:brewing_stand'),
    (118, 'minecraft:cauldron'),
    (119, 'minecraft:end_portal'),
    (120, 'minecraft:end_portal_frame'),
    (121, 'minecraft:end_stone'),
    (122, 'minecraft:dragon_egg'),
    (123, 'minecraft:redstone_lamp'),
    (124, 'minecraft:lit_redstone_lamp'),
    (125, 'minecraft:double_wooden_slab'),
    (126, 'minecraft:wooden_slab'),
    (127, 'minecraft:cocoa'),
    (128, 'minecraft:sandstone_stairs'),
    (129, 'minecraft:emerald_ore'),
    (130, 'minecraft:ender_chest'),
    (131, 'minecraft:tripwire_hook'),
    (132, 'minecraft:tripwire'),
    (133, 'minecraft:emerald_block'),
    (134, 'minecraft:spruce_stairs'),
    (135, 'minecraft:birch_stairs'),
    (136, 'minecraft:jungle_stairs'),
    (137, 'minecraft:command_block'),
    (138, 'minecraft:beacon'),
    (139, 'minecraft:cobblestone_wall'),
    (140, 'minecraft:flower_pot'),
    (141, 'minecraft:carrots'),
    (142, 'minecraft:potatoes'),
    (143, 'minecraft:wooden_button'),
    (144, 'minecraft:skull'),
    (145, 'minecraft:anvil'),
    (146, 'minecraft:trapped_chest'),
    (147, 'minecraft:light_weighted_pressure_plate'),
    (148, 'minecraft:heavy_weighted_pressure_plate'),
    (149, 'minecraft:unpowered_comparator'),
    (150, 'minecraft:powered_comparator'),
    (151, 'minecraft:daylight_detector'),
    (152, 'minecraft:redstone_block'),
    (153, 'minecraft:quartz_ore'),
    (154, 'minecraft:hopper'),
    (155, 'minecraft:quartz_block'),
    (156, 'minecraft:quartz_stairs'),
    (157, 'minecraft:activator_rail'),
    (158, 'minecraft:dropper'),
    (159, 'minecraft:stained_hardened_clay'),
    (160, 'minecraft:stained_glass_pane'),
    (161, 'minecraft:leaves2'),
    (162, 'minecraft:log2'),
    (163, 'minecraft:acacia_stairs'),
    (164, 'minecraft:dark_oak_stairs'),
    (165, 'minecraft:slime'),
    (166, 'minecraft:barrier'),
    (167, 'minecraft:iron_trapdoor'),
    (168, 'minecraft:prismarine'),
    (169, 'minecraft:sea_lantern'),
    (170, 'minecraft:hay_block'),
    (171, 'minecraft:carpet'),
    (172, 'minecraft:hardened_clay'),
    (173, 'minecraft:coal_block'),
    (174, 'minecraft:packed_ice'),
    (175, 'minecraft:double_plant'),
    (176, 'minecraft:standing_banner'),
    (177, 'minecraft:wall_banner'),
    (178, 'minecraft:daylight_detector_inverted'),
    (179, 'minecraft:red_sandstone'),
    (180, 'minecraft:red_sandstone_stairs'),
    (181, 'minecraft:double_stone_slab2'),
    (182, 'minecraft:stone_slab2'),
    (183, 'minecraft:spruce_fence_gate'),
    (184, 'minecraft:birch_fence_gate'),
    (185, 'minecraft:jungle_fence_gate'),
    (186, 'minecraft:dark_oak_fence_gate'),
    (187, 'minecraft:acacia_fence_gate'),
    (188, 'minecraft:spruce_fence'),
    (189, 'minecraft:birch_fence'),
    (190, 'minecraft:jungle_fence'),
    (191, 'minecraft:dark_oak_fence'),
    (192, 'minecraft:acacia_fence'),
    (193, 'minecraft:spruce_door'),
    (194, 'minecraft:birch_door'),
    (195, 'minecraft:jungle_door'),
    (196, 'minecraft:acacia_door'),
    (197, 'minecraft:dark_oak_door'),
    (198, 'minecraft:end_rod'),
    (199, 'minecraft:chorus_plant'),
    (200, 'minecraft:chorus_flower'),
    (201, 'minecraft:purpur_block'),
    (202, 'minecraft:purpur_pillar'),
    (203, 'minecraft:purpur_stairs'),
    (204, 'minecraft:purpur_double_slab'),
    (205, 'minecraft:purpur_slab'),
    (206, 'minecraft:end_bricks'),
    (207, 'minecraft:beetroots'),
    (208, 'minecraft:grass_path'),
    (209, 'minecraft:end_gateway'),
    (210, 'minecraft:repeating_command_block'),
    (211, 'minecraft:chain_command_block'),
    (212, 'minecraft:frosted_ice'),
    (213, 'minecraft:magma'),
    (214, 'minecraft:nether_wart_block'),
    (215, 'minecraft:red_nether_brick'),
    (216, 'minecraft:bone_block'),
    (217, 'minecraft:structure_void'),
    (218, 'minecraft:observer'),
    (219, 'minecraft:white_shulker_box'),
    (220, 'minecraft:orange_shulker_box'),
    (221, 'minecraft:magenta_shulker_box'),
    (222, 'minecraft:light_blue_shulker_box'),
    (223, 'minecraft:yellow_shulker_box'),
    (224, 'minecraft:lime_shulker_box'),
    (225, 'minecraft:pink_shulker_box'),
    (226, 'minecraft:gray_shulker_box'),
    (227, 'minecraft:light_gray_shulker_box'),
    (228, 'minecraft:cyan_shulker_box'),
    (229, 'minecraft:purple_shulker_box'),
    (230, 'minecraft:blue_shulker_box'),
    (231, 'minecraft:brown_shulker_box'),
    (232, 'minecraft:green_shulker_box'),
    (233, 'minecraft:red_shulker_box'),
    (234, 'minecraft:black_shulker_box'),
    (235, 'minecraft:white_glazed_terracotta'),
    (236, 'minecraft:orange_glazed_terracotta'),
    (237, 'minecraft:magenta_glazed_terracotta'),
    (238, 'minecraft:light_blue_glazed_terracotta'),
    (239, 'minecraft:yellow_glazed_terracotta'),
    (240, 'minecraft:lime_glazed_terracotta'),
    (241, 'minecraft:pink_glazed_terracotta'),
    (242, 'minecraft:gray_glazed_terracotta'),
    (243, 'minecraft:light_gray_glazed_terracotta'),
    (244, 'minecraft:cyan_glazed_terracotta'),
    (245, 'minecraft:purple_glazed_terracotta'),
    (246, 'minecraft:blue_glazed_terracotta'),
    (247, 'minecraft:brown_glazed_terracotta'),
    (248, 'minecraft:green_glazed_terracotta'),
    (249, 'minecraft:red_glazed_terracotta'),
    (250, 'minecraft:black_glazed_terracotta'),
    (251, 'minecraft:concrete'),
    (252, 'minecraft:concrete_powder'),
    (253, 'minecraft:structure_block'),

    (255, 'minecraft:iron_shovel'),
    (256, 'minecraft:iron_pickaxe'),
    (257, 'minecraft:iron_axe'),
    (258, 'minecraft:flint_and_steel'),
    (259, 'minecraft:apple'),
    (260, 'minecraft:bow'),
    (261, 'minecraft:arrow'),
    (262, 'minecraft:coal'),
    (263, 'minecraft:diamond'),
    (264, 'minecraft:iron_ingot'),
    (265, 'minecraft:gold_ingot'),
    (266, 'minecraft:iron_sword'),
    (267, 'minecraft:wooden_sword'),
    (268, 'minecraft:wooden_shovel'),
    (269, 'minecraft:wooden_pickaxe'),
    (270, 'minecraft:wooden_axe'),
    (271, 'minecraft:stone_sword'),
    (272, 'minecraft:stone_shovel'),
    (273, 'minecraft:stone_pickaxe'),
    (274, 'minecraft:stone_axe'),
    (275, 'minecraft:diamond_sword'),
    (276, 'minecraft:diamond_shovel'),
    (277, 'minecraft:diamond_pickaxe'),
    (278, 'minecraft:diamond_axe'),
    (279, 'minecraft:stick'),
    (280, 'minecraft:bowl'),
    (281, 'minecraft:mushroom_stew'),
    (282, 'minecraft:golden_sword'),
    (283, 'minecraft:golden_shovel'),
    (284, 'minecraft:golden_pickaxe'),
    (285, 'minecraft:golden_axe'),
    (286, 'minecraft:string'),
    (287, 'minecraft:feather'),
    (288, 'minecraft:gunpowder'),
    (289, 'minecraft:wooden_hoe'),
    (290, 'minecraft:stone_hoe'),
    (291, 'minecraft:iron_hoe'),
    (292, 'minecraft:diamond_hoe'),
    (293, 'minecraft:golden_hoe'),
    (294, 'minecraft:wheat_seeds'),
    (295, 'minecraft:wheat'),
    (296, 'minecraft:bread'),
    (297, 'minecraft:leather_helmet'),
    (298, 'minecraft:leather_chestplate'),
    (299, 'minecraft:leather_leggings'),
    (300, 'minecraft:leather_boots'),
    (301, 'minecraft:chainmail_helmet'),
    (302, 'minecraft:chainmail_chestplate'),
    (303, 'minecraft:chainmail_leggings'),
    (304, 'minecraft:chainmail_boots'),
    (305, 'minecraft:iron_helmet'),
    (306, 'minecraft:iron_chestplate'),
    (307, 'minecraft:iron_leggings'),
    (308, 'minecraft:iron_boots'),
    (309, 'minecraft:diamond_helmet'),
    (310, 'minecraft:diamond_chestplate'),
    (311, 'minecraft:diamond_leggings'),
    (312, 'minecraft:diamond_boots'),
    (313, 'minecraft:golden_helmet'),
    (314, 'minecraft:golden_chestplate'),
    (315, 'minecraft:golden_leggings'),
    (316, 'minecraft:golden_boots'),
    (317, 'minecraft:flint'),
    (318, 'minecraft:porkchop'),
    (319, 'minecraft:cooked_porkchop'),
    (320, 'minecraft:painting'),
    (321, 'minecraft:golden_apple'),
    (322, 'minecraft:sign'),
    (323, 'minecraft:wooden_door'),
    (324, 'minecraft:bucket'),
    (325, 'minecraft:water_bucket'),
    (326, 'minecraft:lava_bucket'),
    (327, 'minecraft:minecart'),
    (328, 'minecraft:saddle'),
    (329, 'minecraft:iron_door'),
    (330, 'minecraft:redstone'),
    (331, 'minecraft:snowball'),
    (332, 'minecraft:boat'),
    (333, 'minecraft:leather'),
    (334, 'minecraft:milk_bucket'),
    (335, 'minecraft:brick'),
    (336, 'minecraft:clay_ball'),
    (337, 'minecraft:reeds'),
    (338, 'minecraft:paper'),
    (339, 'minecraft:book'),
    (340, 'minecraft:slime_ball'),
    (341, 'minecraft:chest_minecart'),
    (342, 'minecraft:furnace_minecart'),
    (343, 'minecraft:egg'),
    (344, 'minecraft:compass'),
    (345, 'minecraft:fishing_rod'),
    (346, 'minecraft:clock'),
    (347, 'minecraft:glowstone_dust'),
    (348, 'minecraft:fish'),
    (349, 'minecraft:cooked_fish'),
    (350, 'minecraft:dye'),
    (351, 'minecraft:bone'),
    (352, 'minecraft:sugar'),
    (353, 'minecraft:cake'),
    (354, 'minecraft:bed'),
    (355, 'minecraft:repeater'),
    (356, 'minecraft:cookie'),
    (357, 'minecraft:filled_map'),
    (358, 'minecraft:shears'),
    (359, 'minecraft:melon'),
    (360, 'minecraft:pumpkin_seeds'),
    (361, 'minecraft:melon_seeds'),
    (362, 'minecraft:beef'),
    (363, 'minecraft:cooked_beef'),
    (364, 'minecraft:chicken'),
    (365, 'minecraft:cooked_chicken'),
    (366, 'minecraft:rotten_flesh'),
    (367, 'minecraft:ender_pearl'),
    (368, 'minecraft:blaze_rod'),
    (369, 'minecraft:ghast_tear'),
    (370, 'minecraft:gold_nugget'),
    (371, 'minecraft:nether_wart'),
    (372, 'minecraft:potion'),
    (373, 'minecraft:glass_bottle'),
    (374, 'minecraft:spider_eye'),
    (375, 'minecraft:fermented_spider_eye'),
    (376, 'minecraft:blaze_powder'),
    (377, 'minecraft:magma_cream'),
    (378, 'minecraft:brewing_stand'),
    (379, 'minecraft:cauldron'),
    (380, 'minecraft:ender_eye'),
    (381, 'minecraft:speckled_melon'),
    (382, 'minecraft:spawn_egg'),
    (383, 'minecraft:experience_bottle'),
    (384, 'minecraft:fire_charge'),
    (385, 'minecraft:writable_book'),
    (386, 'minecraft:written_book'),
    (387, 'minecraft:emerald'),
    (388, 'minecraft:item_frame'),
    (389, 'minecraft:flower_pot'),
    (390, 'minecraft:carrot'),
    (391, 'minecraft:potato'),
    (392, 'minecraft:baked_potato'),
    (393, 'minecraft:poisonous_potato'),
    (394, 'minecraft:map'),
    (395, 'minecraft:golden_carrot'),
    (396, 'minecraft:skull'),
    (397, 'minecraft:carrot_on_a_stick'),
    (398, 'minecraft:nether_star'),
    (399, 'minecraft:pumpkin_pie'),
    (400, 'minecraft:fireworks'),
    (401, 'minecraft:firework_charge'),
    (402, 'minecraft:enchanted_book'),
    (403, 'minecraft:comparator'),
    (404, 'minecraft:netherbrick'),
    (405, 'minecraft:quartz'),
    (406, 'minecraft:tnt_minecart'),
    (407, 'minecraft:hopper_minecart'),
    (408, 'minecraft:prismarine_shard'),
    (409, 'minecraft:prismarine_crystals'),
    (410, 'minecraft:rabbit'),
    (411, 'minecraft:cooked_rabbit'),
    (412, 'minecraft:rabbit_stew'),
    (413, 'minecraft:rabbit_foot'),
    (414, 'minecraft:rabbit_hide'),
    (415, 'minecraft:armor_stand'),
    (416, 'minecraft:iron_horse_armor'),
    (417, 'minecraft:golden_horse_armor'),
    (418, 'minecraft:diamond_horse')
]

stair_ord = [53,67,108,109,114,128,134,135,136,156,163,164,180,203]
#      pos_yaw_list = [(real_h,0,0,9),(0,W,0,18),(0,0,L,18),(0,-W,0,18)] # y,x,z,yaw

def stair_direct(x,z,cx,cz):
    x_dis = abs(x-cx)
    z_dis = abs(z-cz)
    if x>cx and z<=cz:
        if x_dis > z_dis:
            return 1
        else:
            return 2
    elif x<=cx and z<=cz:
        if x_dis > z_dis:
            return 0
        else:
            return 2
    elif x>cx and z>cz:
        if x_dis > z_dis:
            return 1
        else:
            return 3
    elif x<=cx and z>cz:
        if x_dis > z_dis:
            return 0
        else:
            return 3

def add_distance(pos_yaw_list, dis):
    # if dis<=1:
    #     dis = int(1)
    dis -=1
    pos_yaw_list[0][1] -= dis
    pos_yaw_list[0][2] -= dis
    pos_yaw_list[1][1] += 2*dis
    # pos_yaw_list[1][2] -= dis
    # pos_yaw_list[2][1] -= dis
    pos_yaw_list[2][2] += 2*dis
    pos_yaw_list[3][1] -= 2*dis
    # pos_yaw_list[3][2] -= dis
    return pos_yaw_list

START_DATASET_NUM = 30 # [START_DATASET_NUM, DATASET_NUM)
DATASET_NUM = 60

# ORD_PATH = './VoxelAugmented_new/VoxelAugmented_V_auged_ord.npy'
# SAVE_PATH = "VoxelAugmented2Image_V_auged"
# LOAD_NPY_PATH = "./VoxelAugmented_new/VoxelAugmented_V_auged_"

# ORD_PATH = './ISM_V_auged/ISM_V_auged_ord.npy'
# SAVE_PATH = "ISM2Image_V_auged"
ORD_PATH = './ISM_V_auged/ISM_V_auged_ord.npy'
SAVE_PATH = "temptest"
LOAD_NPY_PATH = "./ISM_V_auged/ISM_V_auged_"

if __name__ == "__main__":
    ord_match = np.load(ORD_PATH)
    print(len(ord_match))
    # print(*ord_match)

    task_id_list=[
        # "harvest_1_cobblestone_with_wooden_pickaxe"
        # "combat_spider_plains_leather_armors_diamond_sword_shield",
        # "combat_enderman_end_diamond_armors_diamond_sword_shield",
        # "combat_zombie_plains_diamond_armors_diamond_sword_shield",
        # "combat_zombie_extreme_hills_diamond_armors_iron_sword_shield",
        # "combat_sheep_plains_diamond_armors_iron_sword_shield",
        # "combat_wither_skeleton_nether_leather_armors_diamond_sword_shield",
        # "harvest_wool_with_shears_and_sheep",
        # "harvest_milk_with_empty_bucket_and_cow",
        "harvest_1_dirt",
        # "harvest_1_wheat_swampland",
        # "harvest_1_wheat_forest",
        # "harvest_1_wheat_taiga",
        # "harvest_1_wheat_jungle",
        # "harvest_1_banner_with_crafting_table",
        # "techtree_from_diamond_to_redstone_compass",
    ]
    seed=6+1

    # biome info : BIOMES_MAP
    # block info : https://minecraft-ids.grahamedgecombe.com/

    # biome_id_list = [
    #     # None,
    #     "desert_hills",
    #     # "forest_hills",
    #     # "stone_beach",
    #     # "extreme_hills_m",
    #     # "extreme_hills_plus",
    #     # "extreme_hills_plus_m",
    # ]

    file = np.empty((0,H,W,L))
    # for i in range(DATASET_NUM):
    #     temp = np.load('padded_combined_block_{}.npy'.format(i),allow_pickle = True)
    #     file = np.concatenate((file,temp),0)

    group_ord = 0
    in_group_ord = 0
    save_path="./"
    np.random.seed(seed)
    save_pth = os.path.join(save_path, SAVE_PATH)
    if not os.path.exists(save_pth):
        os.mkdir(save_pth)    
    env = minedojo.make(
        task_id="harvest_1_dirt",
        image_size=(512, 512),
        seed=seed,
        spawn_rate = None,
        allow_time_passage = False,
        # raise_error_on_invalid_cmds = True,
        # generate_world_type = "flat",
        # specified_biome = 127,
        ## set teleport_reset mode
        ## teleport distance: [0,300]
        # fast_reset=True,
        # fast_reset_random_teleport_range_low=0,
        # fast_reset_random_teleport_range_high=300,
    )
    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    env.reset()
    temp_total_num = 1

    for i_ in range(START_DATASET_NUM):
        file = np.empty((0,H,W,L))
        file = np.load(LOAD_NPY_PATH+'{}.npy'.format(i_),allow_pickle = True)
        total_len = file.shape[0]
        for i in range(total_len):
            if i==0 and i_==0:
                continue
            elif ord_match[temp_total_num] == ord_match[temp_total_num-1]:
                in_group_ord+=1
            else:
                group_ord += 1
                # if group_ord != ord_match[temp_total_num]:
                #     print(i_)
                #     print(i)
                #     print(group_ord)
                #     print(ord_match[temp_total_num])
                assert(group_ord == ord_match[temp_total_num])
                in_group_ord = 0
            temp_total_num +=1

    for i_ in range(START_DATASET_NUM,DATASET_NUM):
        file = np.empty((0,H,W,L))
        file = np.load(LOAD_NPY_PATH+'{}.npy'.format(i_),allow_pickle = True)
        # file = np.concatenate((file,temp),0)
        print(file.shape)

        total_len = file.shape[0]

        # RESET_PITCH = 0 # -180
        # RESET_YAW = 0 # -180


        ORI_X =0
        ORI_Y =0
        ORI_Z =0
        ORI_YAW =0
        ORI_PITCH = 0
        TEMP_ORD = 0

        warm_wait_flag = False

        steps = 4
        spectator = f'/gamemode spectator'
        obs, reward, done, info = env.execute_cmd(spectator)
        ORI_X,ORI_Y,ORI_Z = obs["location_stats"]["pos"]
        print(ORI_X,ORI_Y,ORI_Y)
        # ORI_PITCH = obs["location_stats"]["pitch"]
        # ORI_YAW = obs["location_stats"]["yaw"]
        # reset_pos = f'/tp @p {ORI_X} {ORI_Y} {ORI_Z} {ORI_PITCH} {ORI_YAW}'
        reset_pos = f'/tp @p {ORI_X} {ORI_Y} {ORI_Z} '
        
        # warm_wait = 5
        # temp_list = []
        # for j in range(warm_wait):
        #     action = env.action_space.no_op()
        #     env.step(action)
        #     img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
        #     print(img.shape)
        #     temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
        #     pth = os.path.join(save_pth, 'init.gif')
        #     imageio.mimsave(pth, temp_list, duration=1)
        for i in range(total_len):
            print("PROCESS:{}/{}/{}".format(START_DATASET_NUM,i_, DATASET_NUM))
            print("process:{}/{}".format(i, total_len))
            img_list = []
            obs = []
            all_air_flag = True

            # for j in range(30):
            #     np.savetxt('combined_{}.txt'.format(j), file[200][j], fmt = '%d')

            env.execute_cmd(reset_pos)
            real_h = 0
            real_dis = 0
            real_w = 0
            real_l= 0
            real_w_min = -1
            real_l_min = -1

            # for h in range(H):
            #     np.savetxt('combined_{}.txt'.format(h), file[TEMP_ORD][h], fmt = '%d')
            for h in range(H+1):
                empty = f'/fill ~{int(-2*W-1)} ~{h} ~{int(-2*L-1)} ~{int(2*W+1)} ~{h} ~{int(2*L+1)} minecraft:air'
                env.execute_cmd(empty)
            layer_block_num = 0
            delta_h = 0
            layer_decrease_flag = True
            for h in range(H):
                h_flag = False
                temp_real_w = 0
                
                if layer_block_num<=4 and h>0 and layer_decrease_flag:
                    delta_h += 1
                    empty = f'/fill ~{int(-1)} ~{h-1} ~{int(-1)} ~{int(W+1)} ~{h-1} ~{int(L+1)} minecraft:air'
                    env.execute_cmd(empty)
                if layer_block_num>5:
                    layer_decrease_flag = False
                if WEATHER_CLEAR:
                    w_c = f'/weather clear'
                    env.execute_cmd(w_c)
                for w in range(W):
                    w_flag = False
                    temp_real_l = 0
                    for l in range(L):
                        # if i!=0 and file[TEMP_ORD+i-1][h][w][l]>0 and file[TEMP_ORD+i][h][w][l] ==0:
                        #     empty = f'/setblock ~{w+1} ~{h} ~{l+1} minecraft:air'
                        #     env.execute_cmd(empty)
                        if file[TEMP_ORD+i][h][w][l] > 0:
                            layer_block_num +=1
                            all_air_flag = False
                            h_flag = True
                            w_flag = True
                            name = minecraft_items[int(file[TEMP_ORD+i][h][w][l])][1]
                            # if i==0 or file[TEMP_ORD+i][h][w][l] != file[TEMP_ORD+i-1][h][w][l]:
                            build = f'/setblock ~{w+1} ~{h-delta_h} ~{l+1} '+name
                            if file[TEMP_ORD+i][h][w][l] in stair_ord:
                                center_x = W/2
                                center_z = L/2
                                s_dir = stair_direct(w,l,center_x,center_z)
                                build = f'/setblock ~{w+1} ~{h-delta_h} ~{l+1} '+name + f' {s_dir}'
                            env.execute_cmd(build)
                            temp_real_l = l
                            if real_l_min == -1:
                                real_l_min = l
                            
                                
                    if w_flag and real_w_min == -1:
                        real_w_min = w
                    elif w_flag:
                        temp_real_w = w
                    if(real_l_min>=0):
                        real_l = max(real_l, temp_real_l - real_l_min)
                        real_l_min = -1
                time_set = f'/time set 6000'
                env.execute_cmd(time_set)

                kill = f'/kill @e[type=!Player]'
                env.execute_cmd(kill)
                if h_flag:
                    real_h = real_h+1
                if real_w_min>=0:
                    real_w = max(real_w, temp_real_w - real_w_min)
                    real_w_min = -1

            # print("real_l:{}".format(real_l))
            # print("real_w:{}".format(real_w))
            real_width = math.sqrt(real_w*real_w + real_l* real_l)
            print("real_width:{}".format(real_width))
            if all_air_flag or real_width == 0:
                all_air_flag = True
                print("ALL ZERO BLOCK!")
                continue

            fit_dis = real_width / math.tan(math.pi * (45/180))
            fit_pitch = math.atan(real_h*3/4/fit_dis)/math.pi * 180
            fit_pitch_num = math.ceil(fit_pitch / 15)
            print("fit_pitch:{}".format(fit_pitch))
            # print("fit_pitch_num:{}".format(fit_pitch_num))

            fit_dis = fit_dis / math.sqrt(2)
            # print("fit_dis:{}".format(fit_dis))

            # for j in range(total_len):
            #     build = f'/setblock ~{_k} ~ ~{_k} minecraft:red_flower'

            # set flowers
            # for _k in range(10):
            #     flower = f'/setblock ~{_k} ~ ~{_k} minecraft:red_flower'
            #     env.execute_cmd(flower)
            pitch_list = [12,12,12,12]
            pitch_list[0]+=fit_pitch_num
            # pitch_list = [[14,12,12,12],[15,12,12,12],[16,12,12,12],[17,12,12,12]] #-60, -45, -30, -15
            

            print("real_h:{}".format(real_h))
            POS_YAW_LIST = [[int(real_h),0,0,9],[0,W,0,18],[0,0,L,18],[0,-W,0,18]] # y,x,z,yaw
            pos_yaw_list = add_distance(POS_YAW_LIST, int(fit_dis-(W+L)/4))

            support = f'/setblock ~ ~{-1} ~ minecraft:glass'
            remove_support = f'/fill ~ ~{-1} ~ ~ ~{-2} ~ minecraft:air'

            segment = 1

            warm_wait = 30
            temp_list = []
            # for j in range(warm_wait):
            #     action = env.action_space.no_op()
            #     obs, reward, done, info = env.step(action)
            #     img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
            #     print(img.shape)
            #     temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))

                # env.reset()

            for j in range(steps*segment):
                action = env.action_space.no_op()

                # Action Space : https://docs.minedojo.org/sections/core_api/action_space.html
                done = False
                if not warm_wait_flag:
                    tp = f'/tp @p ~{pos_yaw_list[0][1]} ~{pos_yaw_list[0][0]} ~{pos_yaw_list[0][2]} '
                    env.execute_cmd(tp)
                    env.execute_cmd(support)
                    for j_ in range(warm_wait):
                        obs, reward, done, info = env.step(action)
                        img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                        # print(img.shape)
                        temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
                    env.execute_cmd(remove_support)
                    
                    tp = f'/tp @p ~{-pos_yaw_list[0][1]} ~{-pos_yaw_list[0][0]} ~{-pos_yaw_list[0][2]} '
                    env.execute_cmd(tp)
                    warm_wait_flag = True

                if j%segment == 0 :
                    k = int(j / segment)
                    dxyz = random_xyz()
                    # tp = f'/tp @p ~{pos_yaw_list[k][1]} ~{pos_yaw_list[k][0]} ~{pos_yaw_list[k][2]} facing ~{int(ORI_X+W/2)} ~{int(ORI_Y + real_h/4)} ~{int(ORI_Z + L/2)}'
                    tp = f'/tp @p ~{pos_yaw_list[k][1]+dxyz[0]} ~{pos_yaw_list[k][0]+dxyz[1]} ~{pos_yaw_list[k][2]+dxyz[2]} '

                    env.execute_cmd(tp)
                    # env.execute_cmd(support)

                    action[3] = pitch_list[k]
                    action[4] = pos_yaw_list[k][3]
                    
                    obs, reward, done, info = env.step(action)
                    # img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                    # print(img.shape)
                    # img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
                    # action[4]=13 # simply turn around

                    # action[3] = RESET_PITCH
                    # action[4] = RESET_YAW
                    
                # if j%segment ==segment-1:
                    action = env.action_space.no_op()
                    # env.execute_cmd(remove_support)
                    obs, reward, done, info = env.step(action)
                    img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                    # print(img.shape)
                    img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
                    
                    
                # else:
                #     # action[4] = 13
                #     obs, reward, done, info = env.step(action)
                #     img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                #     print(img.shape)
                #     temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))

                if done:
                    break
            
            if i==0 and i_==0:
                temp_total_num +=1
            elif ord_match[temp_total_num] == ord_match[temp_total_num-1]:
                in_group_ord+=1
                temp_total_num +=1
            else:
                group_ord += 1
                # if group_ord != ord_match[temp_total_num]:
                #     print(i_)
                #     print(i)
                #     print(group_ord)
                #     print(ord_match[temp_total_num])
                if group_ord!=ord_match[temp_total_num]:
                    print(group_ord)
                    print(temp_total_num)
                    print(ord_match[temp_total_num])
                    print(ord_match[temp_total_num-1])
                    print(ord_match)
                assert(group_ord == ord_match[temp_total_num])
                in_group_ord = 0
                temp_total_num +=1
            
            pth = os.path.join(save_pth, '{}_{}.gif'.format(group_ord,in_group_ord))
            imageio.mimsave(pth, img_list, duration=1)
            # pth = os.path.join(save_pth, 'init{}.gif'.format(i))
            # imageio.mimsave(pth, temp_list, duration=0.5)
            env.reset()
            ## stop fast reset:
            # env.reset(move_flag = False)
    env.close()

    print("[INFO] Test Success")

def setup_env(seed = 7, image_H = 512, image_W = 512):
    env = minedojo.make(
        task_id="harvest_1_dirt",
        image_size=(image_H, image_W),
        seed=seed,
        spawn_rate = None,
        allow_time_passage = False,
        # raise_error_on_invalid_cmds = True,
        # generate_world_type = "flat",
        # specified_biome = 127,
        ## set teleport_reset mode
        ## teleport distance: [0,300]
        # fast_reset=True,
        # fast_reset_random_teleport_range_low=0,
        # fast_reset_random_teleport_range_high=300,
    )
    print(f"[INFO] Create a task with prompt: {env.task_prompt}")

    env.reset()
    return env

def gen_MC_gif(VoxData, minedojo_env, gif_path,save_dir, idx ,reset_pos = (881.5, 4.0, 4.0) , warm_wait_flag = False, add_noise = True, gif_duration = 1,VH=10,VW=32,VL=32, noise_flag = True):

    img_list = []
    obs = []
    all_air_flag = True
    VoxData = remove_noise(VoxData)

    # for j in range(30):
    #     np.savetxt('combined_{}.txt'.format(j), VoxData[j], fmt = '%d')
    reset_pos = f'/tp @p {reset_pos[0]} {reset_pos[1]} {reset_pos[2]} '
    spectator = f'/gamemode spectator'
    obs, reward, done, info = minedojo_env.execute_cmd(spectator)

    minedojo_env.execute_cmd(reset_pos)
    real_h = 0
    real_dis = 0
    real_w = 0
    real_l= 0
    real_w_min = -1
    real_l_min = -1

    for h in range(4,VH+10):
        empty = f'/fill ~{int(-2*VW-1)} {h} ~{int(-2*VL-1)} ~{int(2*VW+1)} {h} ~{int(2*VL+1)} minecraft:air'
        minedojo_env.execute_cmd(empty)
        # print("clear")
    layer_block_num = 0
    delta_h = 0
    layer_decrease_flag = False
    for h in range(VH):
        h_flag = False
        temp_real_w = 0
        
        if layer_block_num<=4 and h>0 and layer_decrease_flag:
            delta_h += 1
            empty = f'/fill ~{int(-1)} ~{h-1} ~{int(-1)} ~{int(VW+1)} ~{h-1} ~{int(VL+1)} minecraft:air'
            minedojo_env.execute_cmd(empty)
        if layer_block_num>5:
            layer_decrease_flag = False
        if WEATHER_CLEAR:
            w_c = f'/weather clear'
            minedojo_env.execute_cmd(w_c)
        for w in range(VW):
            w_flag = False
            temp_real_l = 0
            for l in range(VL):
                if VoxData[h][w][l] > 0:
                    layer_block_num +=1
                    all_air_flag = False
                    h_flag = True
                    w_flag = True
                    name = minecraft_items[int(VoxData[h][w][l])][1]
                    build = f'/setblock ~{w+1} ~{h-delta_h} ~{l+1} '+name
                    if VoxData[h][w][l] in stair_ord:
                        center_x = VW/2
                        center_z = VL/2
                        s_dir = stair_direct(w,l,center_x,center_z)
                        build = f'/setblock ~{w+1} ~{h-delta_h} ~{l+1} '+name + f' {s_dir}'
                    # if not( w+l>32 and noise_flag):
                    obs, reward, done, info = minedojo_env.execute_cmd(build)
                    # img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
                    temp_real_l = l
                    if real_l_min == -1:
                        real_l_min = l
                    
                        
            if w_flag and real_w_min == -1:
                real_w_min = w
            elif w_flag:
                temp_real_w = w
            if(real_l_min>=0):
                real_l = max(real_l, temp_real_l - real_l_min)
                real_l_min = -1
        time_set = f'/time set 6000'
        minedojo_env.execute_cmd(time_set)

        kill = f'/kill @e[type=!Player]'
        minedojo_env.execute_cmd(kill)
        if h_flag:
            real_h = real_h+1
        if real_w_min>=0:
            real_w = max(real_w, temp_real_w - real_w_min)
            real_w_min = -1

    # print("real_l:{}".format(real_l))
    # print("real_w:{}".format(real_w))
    real_width = math.sqrt(real_w*real_w + real_l* real_l)
    print("real_width:{}".format(real_width))
    if all_air_flag or real_width == 0:
        all_air_flag = True
        print("ALL ZERO BLOCK!")
        return -1

    fit_dis = real_width / math.tan(math.pi * (40/180))
    fit_pitch = math.atan(real_h/2/fit_dis)/math.pi * 180
    fit_pitch_num = math.ceil(fit_pitch / 15)
    print("fit_pitch:{}".format(fit_pitch))
    # print("fit_pitch_num:{}".format(fit_pitch_num))

    fit_dis = fit_dis / math.sqrt(2)
    # print("fit_dis:{}".format(fit_dis))

    # for j in range(total_len):
    #     build = f'/setblock ~{_k} ~ ~{_k} minecraft:red_flower'

    # set flowers
    # for _k in range(10):
    #     flower = f'/setblock ~{_k} ~ ~{_k} minecraft:red_flower'
    #     minedojo_env.execute_cmd(flower)
    pitch_list = [12,12,12,12]
    pitch_list[0]+=fit_pitch_num
    # pitch_list = [[14,12,12,12],[15,12,12,12],[16,12,12,12],[17,12,12,12]] #-60, -45, -30, -15
    

    print("real_h:{}".format(real_h))
    POS_YAW_LIST = [[int(real_h),0,0,9],[0,VW,0,18],[0,0,VL,18],[0,-VW,0,18]] # y,x,z,yaw
    pos_yaw_list = add_distance(POS_YAW_LIST, int(fit_dis-(VW+VL)/4))

    support = f'/setblock ~ ~{-1} ~ minecraft:glass'
    remove_support = f'/fill ~ ~{-1} ~ ~ ~{-2} ~ minecraft:air'

    segment = 1

    warm_wait = 30
    temp_list = []
    # for j in range(warm_wait):
    #     action = minedojo_env.action_space.no_op()
    #     obs, reward, done, info = minedojo_env.step(action)
    #     img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
    #     print(img.shape)
    #     temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))

        # minedojo_env.reset()
    WAIT_TIME = 1

    for j in range(4*segment):
        action = minedojo_env.action_space.no_op()

        # Action Space : https://docs.minedojo.org/sections/core_api/action_space.html
        done = False
        if not warm_wait_flag:
            tp = f'/tp @p ~{pos_yaw_list[0][1]} ~{pos_yaw_list[0][0]} ~{pos_yaw_list[0][2]} '
            minedojo_env.execute_cmd(tp)
            minedojo_env.execute_cmd(support)
            for j_ in range(warm_wait):
                obs, reward, done, info = minedojo_env.step(action)
                img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
                # print(img.shape)
                temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
            minedojo_env.execute_cmd(remove_support)
            
            tp = f'/tp @p ~{-pos_yaw_list[0][1]} ~{-pos_yaw_list[0][0]} ~{-pos_yaw_list[0][2]} '
            minedojo_env.execute_cmd(tp)
            warm_wait_flag = True

        if j%segment == 0 :
            k = int(j / segment)
            if add_noise:
                dxyz = random_xyz()
            else :
                dxyz = [0,0,0]
            # tp = f'/tp @p ~{pos_yaw_list[k][1]} ~{pos_yaw_list[k][0]} ~{pos_yaw_list[k][2]} facing ~{int(ORI_X+VW/2)} ~{int(ORI_Y + real_h/4)} ~{int(ORI_Z + VL/2)}'
            tp = f'/tp @p ~{pos_yaw_list[k][1]+dxyz[0]} ~{pos_yaw_list[k][0]+dxyz[1]} ~{pos_yaw_list[k][2]+dxyz[2]} '

            minedojo_env.execute_cmd(tp)
            # minedojo_env.execute_cmd(support)

            action[3] = pitch_list[k]
            action[4] = pos_yaw_list[k][3]
            
            obs, reward, done, info = minedojo_env.step(action)
            
            # img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
            # print(img.shape)
            # img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
            # action[4]=13 # simply turn around

            # action[3] = RESET_PITCH
            # action[4] = RESET_YAW
            
        # if j%segment ==segment-1:
            # minedojo_env.execute_cmd(remove_support)
            for k in range(WAIT_TIME):
                action = minedojo_env.action_space.no_op()
                obs, reward, done, info = minedojo_env.step(action)
            # img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
            # print(img.shape)
            img_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))
            
            
        # else:
        #     # action[4] = 13
        #     obs, reward, done, info = minedojo_env.step(action)
        #     img = np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8)
        #     print(img.shape)
        #     temp_list.append(np.transpose(obs['rgb'], [1,2,0]).astype(np.uint8))

        if done:
            break
    
    # if i==0 and i_==0:
    #     temp_total_num +=1
    # elif ord_match[temp_total_num] == ord_match[temp_total_num-1]:
    #     in_group_ord+=1
    #     temp_total_num +=1
    # else:
    #     group_ord += 1
    #     # if group_ord != ord_match[temp_total_num]:
    #     #     print(i_)
    #     #     print(i)
    #     #     print(group_ord)
    #     #     print(ord_match[temp_total_num])
    #     assert(group_ord == ord_match[temp_total_num])
    #     in_group_ord = 0
    #     temp_total_num +=1
    
    # pth = os.path.join(save_pth, '{}_{}.gif'.format(group_ord,in_group_ord))
    imageio.mimsave(gif_path, img_list, duration=gif_duration)
    temp_len = len(img_list)
    for i in range(temp_len):
        imageio.imsave(os.path.join(save_dir,"{}_{}.jpg".format(idx,i)),img_list[i])
    minedojo_env.reset()
    return 0

def gen_MC_vox(VoxData, save_dir, info="", add_noise = True,VH=10,VW=32,VL=32, noise_flag = True):
    VoxData = remove_noise(VoxData)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    np.save(os.path.join(save_dir,"{}.npy".format(info)),VoxData)
    
