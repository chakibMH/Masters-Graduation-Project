from binarytree import Node


def insert_BST(root, z):
    z = Node(z)
    y = None
    x = root
    while x != None:
        y = x
        if z.value < x.value:
            x = x.left
        else:
            x = x.right
    
    if y == None:
        root = z
    elif z.value < y.value:
        y.left = z
    else:
        y.right = z
    
    return root

def recursive_Tree_Search(x, key):
    
    if x == None or key == x.value:
        return x
    
    if key < x.value:
        return recursive_Tree_Search(x.left, key)
    else:
        return recursive_Tree_Search(x.right, key)
    
    
### exemple

li = ['abstract', 'document', 'search', 'alter']

# always init the root with None

r = None

for elt in li:
    
    r = insert_BST(r, elt)
    

recursive_Tree_Search(r,'alter')
    
    