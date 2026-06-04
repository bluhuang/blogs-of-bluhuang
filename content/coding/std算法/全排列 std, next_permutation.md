---
title: "全排列 std, next_permutation"
date: 2026-04-20
---

# 1 C++官方文档
https://en.cppreference.com/w/cpp/algorithm/next_permutation.html

# 2 原理

## 2.1 一句话总结原理
从右向左找到第一个可以变大的元素的位置，将其替换为右边比它大的最小数字，然后将右边剩余部分重新排列为最小顺序。

`next_permutation` 算法的核心就是**三步**：

1. 从右至左找到第一个升序对（确定需要变动的位）。
2. 从右至左找到第一个大于该位的数（确定交换的位）。
3. 交换，然后反转后面的部分。

## 2.2 原理拆解
### 图解例子：`[1, 3, 5, 4, 2]`

| 步骤 | 数组状态 | 说明 |
|------|----------|------|
| 初始 | `[1, 3, 5, 4, 2]` | |
| **1. 找 i** | 从右向左找 `nums[i] < nums[i+1]`：`2` 无右；`4>2` 不成立；`5>4` 不成立；`3<5` ✅，所以 `i = 1`（指向 `3`） | 找到第一个可增大的位置 i=1，【元素3右边的数已经是降序，无法再变大】 |
| **2. 找 j** | 在 i 右边 `[5,4,2]` 中从右向左找第一个大于 3 的数：`2` 不大于；`4` 大于，所以 `j = 3`（指向 `4`） | 找到刚好比 3 大的数 4 |
| **3. 交换** | 交换 `nums[1]` 和 `nums[3]` → `[1, 4, 5, 3, 2]` | |
| **4. 反转** | 将 i+1（索引2）到末尾 `[5,3,2]` 反转 → `[2,3,5]` | 最终 `[1,4,2,3,5]` |

结果 `14235` 确实是 `13542` 的下一个排列。

## 为什么最后可以反转？
因为交换的两个元素，就是找到的正好比原 i 位置元素大一点的数。交换后，元素3（j位置）的左边都比它大（否则不会交换j），右边都比它小（因为是找到第一个比3大的元素进行的交换），因此交换后右边的这些元素依然保持降序，反转后会变成升序。

## 为什么步骤必须这样？

- **找 i**：保证改动的是尽可能靠右的位置，这样整体增大最小。
- **找 j**：保证换到 i 位置的数是右边**最小的较大数**，这样增大刚好一点点。
- **反转**：因为 i 右边原本是降序（最大顺序），现在 i 变大后，右边需要变成升序（最小顺序）才能得到刚好大的下一个排列。

# 3 代码实现（简洁版）

```cpp
bool nextPermutation(vector<int>& nums) {
    int n = nums.size();
    int i = n - 2;
    while (i >= 0 && nums[i] >= nums[i + 1]) i--;
    if (i >= 0) {
        int j = n - 1;
        while (nums[j] <= nums[i]) j--;
        swap(nums[i], nums[j]);
    }
    reverse(nums.begin() + i + 1, nums.end());
    return i >= 0;
}
```

注意使用 `>=` 和 `<=` 是为了跳过相等元素，确保严格递增/递减。

# 4 用法

## 头文件

```cpp
#include <algorithm>
```

## 函数原型

```cpp
// 使用 operator< 比较元素
bool next_permutation (BidirectionalIterator first, BidirectionalIterator last);
// 使用自定义比较函数 comp
bool next_permutation (BidirectionalIterator first, BidirectionalIterator last, Compare comp);
```

## 作用

将序列 `[first, last)` 重新排列为**字典序中的下一个排列**。

- 如果存在下一个排列（即当前排列不是最大排列），则将其转换为下一个排列，并返回 `true`。
- 如果当前排列已经是字典序最大的排列（即完全降序），则将其转换为最小的排列（即完全升序），并返回 `false`。

## 示例代码：生成全排列

```cpp
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

int main() {
    vector<int> nums = {1, 2, 3};  // 初始为升序（最小排列）

    do {
        for (int x : nums) cout << x << ' ';
        cout << '\n';
    } while (next_permutation(nums.begin(), nums.end()));

    return 0;
}
```

## 自定义比较函数

例如，对字符串按长度排序（长度小的在前，相同长度按字典序）：

```cpp
vector<string> words = {"apple", "banana", "cherry"};
auto comp = [](const string& a, const string& b) {
    return a.length() < b.length();   // 按长度升序
};
sort(words.begin(), words.end(), comp);  // 先排序
do {
    for (const auto& w : words) cout << w << ' ';
    cout << '\n';
} while (next_permutation(words.begin(), words.end(), comp));
```