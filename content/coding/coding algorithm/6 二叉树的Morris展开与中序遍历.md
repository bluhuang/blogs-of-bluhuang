---
title: "二叉树的Morris展开与中序遍历"
date: 2026-04-20
---

### 一、什么是 Morris 遍历？

Morris 遍历是一种**使用线索二叉树（threaded binary tree）思想**的二叉树遍历算法。它的**核心特点**是：
- **空间复杂度 O(1)**：不使用递归栈或队列，仅利用二叉树中的空闲指针（右指针）临时记录遍历的后继节点，从而完成遍历。
- **时间复杂度 O(n)**：每个节点最多被访问两次。
Morris 遍历可适用于**前序、中序、后序**遍历，其中**中序遍历**最为经典和常用。

### 二、为什么可以用 Morris 遍历？

在常规的递归或迭代遍历中，需要额外的栈来保存返回路径，空间复杂度为 O(h)（h 为树高）。Morris 遍历巧妙地利用叶子节点的空右指针，将其指向中序遍历下的后继节点，从而在遍历过程中无需栈便可回到上层节点。访问完毕后，再恢复空指针，确保树的结构不受影响。

**适用场景**：
- 需要在 O(1) 空间内遍历或修改二叉树。
- 允许临时修改树的结构。
- 不能使用递归或栈（递归的空间开销为 O(h)）。

### 三、Morris 遍历的核心思想

#### 1. 将树线性展开
- 从根节点开始，当前节点记为 `cur`。
- 如果 `cur` 有左子树：
    - 找到左子树的最右节点 `pre`（即左子树中一直向右走到尽头）。
    - 将 `cur` 的右子树接到 `pre` 的右边。
    - 将 `cur` 的左子树移到右边：`cur->right = cur->left`，`cur->left = nullptr`。
- 然后 `cur` 移至下一个右节点（`cur = cur->right`），重复此过程。

```cpp
// 二叉树展开为链表
class Solution {
public:
    void flatten(TreeNode* root) {
        TreeNode* cur = root;
        while (cur) {
            if (cur->left) {
                // 找到左子树的最右节点
                TreeNode* pre = cur->left;
                while (pre->right) pre = pre->right;
                // 将原右子树接到最右节点
                pre->right = cur->right;
                // 将左子树移到右边
                cur->right = cur->left;
                cur->left = nullptr;
            }
            cur = cur->right;
        }
    }
};
```

#### 2. 中序遍历（经典）
### 核心思想
假设你走在一个迷宫（二叉树）中，每次遇到左岔路时，你想先走完左边再返回。但你没有栈（无法记路），怎么办？  
你可以这样：在进入左边之前，于左路最深处（最右边的节点）放一根绳子，绳子另一头系在当前节点上。走完左边后，顺着绳子即可返回。返回后再解开绳子，然后继续向右走。  
这就是 Morris 遍历的本质：利用叶子节点的空右指针作为“绳子”（线索），临时指向后继节点，从而在 O(1) 空间内完成遍历。

中序遍历的 Morris 算法步骤：
1. 初始化当前节点 `cur` 为根节点。
2. 如果 `cur` 没有左孩子：
    - 访问 `cur`。
    - `cur = cur->right`（移至右子树）。
3. 如果 `cur` 有左孩子：
    - 找到左子树中最右的节点（即中序遍历下 `cur` 的前驱节点 `pre`）。
    - 如果 `pre` 的右指针为空，说明尚未建立线索，则将 `pre->right` 指向 `cur`（建立线索），然后 `cur = cur->left`。
    - 如果 `pre` 的右指针已指向 `cur`，说明左子树已遍历完，此时需恢复：将 `pre->right` 置空，访问 `cur`，然后 `cur = cur->right`。

```cpp
// 二叉树的中序遍历
vector<int> inorderTraversal(TreeNode* root) {
    vector<int> res;
    TreeNode* cur = root;
    while (cur) {
        if (!cur->left) {
            // 访问当前节点
            res.push_back(cur->val);
            cur = cur->right;
        } else {
            // 找左子树的最右节点
            TreeNode* pre = cur->left;
            while (pre->right && pre->right != cur) pre = pre->right;
            if (!pre->right) {
                // 建立绳索，指向 cur
                pre->right = cur;
                cur = cur->left;
            } else {
                // 已建立线索，说明左子树遍历完，恢复指针
                pre->right = nullptr;
                // 访问当前节点
                res.push_back(cur->val);
                cur = cur->right;
            }
        }
    }
    return res;
}
```