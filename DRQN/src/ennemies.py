import numpy as np
ENNEMIES = [
    "Arachnotron", "Archvile", "BaronOfHell", "HellKnight", "Cacodemon", "Cyberdemon",
    "Demon", "Spectre", "ChaingunGuy", "DoomImp", "Fatso", "LostSoul", "PainElemental", "Revenant",
    "ShotgunGuy", "SpiderMastermind", "WolfensteinSS", "ZombieMan", "MarineBFG", "MarineBerserk",
    "MarineChaingun", "MarineChainsaw", "MarineFist", "MarinePistol", "MarinePlasma", "MarineRailgun",
    "MarineRocket", "MarineSSG", "MarineShotgun", "ScriptedMarine", "StealthArachnotron",
    "StealthArchvile", "StealthBaron", "StealthHellKnight", "StealthCacodemon", "StealthDemon",
    "StealthChaingunGuy", "StealthDoomImp", "StealthFatso", "StealthRevenant", "StealthShotgunGuy",
    "StealthZombieMan", "Zombieman",
]

PICKUPS = [
    "Allmap", "ArmorBonus", "Backpack", "Berserk", "BFG9000", "BlueArmor", "BlueCard",
    "BlueSkull", "BlurSphere", "Cell", "CellPack", "Chaingun", "Chainsaw", "Clip", "ClipBox",
    "Fist", "GreenArmor", "HealthBonus", "Infrared", "InvulnerabilitySphere", "Medikit",
    "Megasphere", "Pistol", "PlasmaRifle", "RadSuit", "RedCard", "RedSkull", "RocketAmmo",
    "RocketBox", "RocketLauncher", "Shell", "ShellBox", "Shotgun", "Soulsphere", "Stimpack",
    "SuperShotgun", "YellowCard", "YellowSkull"
]

BLASTS = [
    "BulletPuff", "Blood", "BaronBall"
]

IGNORABLE = [
    "TeleportFog", "DoomPlayer"
]


def get_cone_entities(state, entity_type):
    return [x for x in state.labels if x.object_name in entity_type]


def has_visible(state, walls, entity_type):
    cone = get_cone_entities(state, entity_type)
    player = (state.game_variables[0], state.game_variables[1])
    for entity in cone:
        if all([is_visible(player, wall, entity) for wall in walls]):
            return True
    return False

def min_relative_pos(state, entity_type):
    relative_x = 0
    relative_y = 0
    min_dist = float('inf')
    for obj in state.labels:
        # obj_pos_x = obj.object_position_x
        # obj_pos_y = obj.object_position_y
        if(obj.object_name not in entity_type):
            continue
        a = np.array((obj.object_position_x ,obj.object_position_y))
        b = np.array((state.game_variables[0], state.game_variables[1]))
        # print(type(obj.object_name))
        if(obj.object_name != 'DoomPlayer'):
            dist = np.linalg.norm(a-b)
            if(dist < min_dist):
                min_dist = dist
                relative_x = obj.object_position_x - state.game_variables[0]
                relative_y = obj.object_position_y - state.game_variables[1]

    # return [relative_x, relative_y]
    if(relative_x == 0 and relative_y == 0):
        return 0
    else:
        return np.tanh(np.divide(relative_x, relative_y))



def get_min_relative_pos(state):
    types = ENNEMIES, PICKUPS, BLASTS
    return [min_relative_pos(state, x) for x in types]


def has_visible_entities(state, wall):
    types = ENNEMIES, PICKUPS, BLASTS
    return [has_visible(state, wall, x) for x in types]

def get_game_feature(state, wall):
    types = ENNEMIES, PICKUPS, BLASTS
    result = []
    for x in types:
        result.append(has_visible(state, wall, x))
        # result.append(min_relative_pos(state, x)[0])
        # result.append(min_relative_pos(state, x)[1])
        result.append(min_relative_pos(state, x))
    return result


def is_visible(player, wall, entity):
    a = player
    b = (entity.object_position_x, entity.object_position_y)
    c, d = wall

    return not does_intersect(a, b, c, d)


def ccw(a, b, c):
    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


def does_intersect(a, b, c, d):
    "return true if line segments ab and cd intersect"
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
