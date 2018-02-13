# @Time    : 2018/2/8 17:01
# @Author  : myailab
# @Site    : www.myailab.cn
# @File    : __init__.py
# @Software: PyCharm

import chapter12_fpgrowth.fpGrowth as fpg

if __name__ == '__main__':
    # method = "show_node"
    # method = "fpg_test"
    # method = "get_cpb"  # 抽取条件模式基(conditional pattern base)
    method = "condition_fpg"  # 创建条件FP树
    # method = "stagewise"

    # data_matrix, class_labels = ada.load_simple_data()

    if method == "show_node":
        root_node = fpg.treeNode('pyramid', 9, None)
        root_node.children['eye'] = fpg.treeNode('eye', 13, None)
        root_node.children['phoenix'] = fpg.treeNode('phoenix', 3, None)
        root_node.display()
    elif method == 'fpg_test':
        simple_data = fpg.load_simple_data()
        init_set = fpg.createInitSet(simple_data)
        my_fp_tree, my_header_table = fpg.createTree(init_set, 3)
        my_fp_tree.display()
    elif method == 'get_cpb':
        simple_data = fpg.load_simple_data()
        init_set = fpg.createInitSet(simple_data)
        my_fp_tree, my_header_table = fpg.createTree(init_set, 3)
        x_result = fpg.findPrefixPath('x', my_header_table['x'][1])
        z_result = fpg.findPrefixPath('z', my_header_table['z'][1])
        r_result = fpg.findPrefixPath('r', my_header_table['r'][1])
        print(x_result)
        print(z_result)
        print(r_result)
    elif method == 'condition_fpg':
        frequent_item = []
        simple_data = fpg.load_simple_data()
        init_set = fpg.createInitSet(simple_data)
        my_fp_tree, my_header_table = fpg.createTree(init_set, 3)
        fpg.mineTree(my_fp_tree, my_header_table, 3, set([]), frequent_item)
        print("frequent_item:")
        print(frequent_item)


