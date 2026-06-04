---
title: "Application-Defined SQL Functions"
categories: ["database"]
author: "BluHuang"
date: 2026-06-04T19:01:05+0800
lastmod: 2025-12-30
---

[[programming interfaces]]

source: https://sqlite.org/appfunc.html
# 1 简介
1. SQLite 支持用户实现自定义回调方法，从而实现特定的扩展功能，类似于用户自定义函数（UDF）。
2. 自定义的 SQLite UDF 可以是标量函数、聚合函数，也可以是窗口函数。
3. 用户可以创建可在 SQL 查询中使用的自定义函数，这些函数可用 C 语言编写，也可以通过绑定在其他语言（如 Python、Java 等）中实现。

# 2 相关接口

## sqlite3_create_function 系列

### 1 sqlite3_create_function()
#### 基本信息

```
int sqlite3_create_function(
  sqlite3 *db,            /* 数据库连接 */
  const char *zFuncName,  /* 函数名 (UTF-8) */
  int nArg,               /* 参数个数 (-1 = 可变) */
  int eTextRep,           /* 首选文本编码 */
  void *pApp,             /* 应用数据指针 */
  void (*xFunc)(sqlite3_context*,int,sqlite3_value**), /* 标量函数 */
  void (*xStep)(sqlite3_context*,int,sqlite3_value**), /* 聚合 step 函数 */
  void (*xFinal)(sqlite3_context*)                     /* 聚合 finalize 函数 */
);
```

#### 特点

- **最基础版本**：最早引入的函数注册接口。
- **编码指定**：使用 `eTextRep` 参数指定文本编码偏好。
- **函数类型**：
    - 若 `xFunc` 非 NULL，则注册标量函数；
    - 若 `xStep` 和 `xFinal` 非 NULL，则注册聚合函数。
- **应用数据**：通过 `pApp` 传递用户数据。
- **内存管理**：无自动清理机制，需手动管理内存。

### 2 sqlite3_create_function16()

#### 基本信息
```
int sqlite3_create_function16(
  sqlite3 *db,           /* 数据库连接 */
  const void *zFuncName, /* 函数名 (UTF-16) */
  int nArg,              /* 参数个数 */
  int eTextRep,          /* 首选文本编码 */
  void *pApp,            /* 应用数据指针 */
  void (*xFunc)(sqlite3_context*,int,sqlite3_value**),
  void (*xStep)(sqlite3_context*,int,sqlite3_value**),
  void (*xFinal)(sqlite3_context*)
);
```

#### 特点
- **UTF-16 支持**：函数名使用 UTF-16 编码。
- **向后兼容**：为需要 UTF-16 函数名的应用提供支持。
- **功能相同**：与 `sqlite3_create_function()` 功能相同，仅函数名编码不同。
- **使用场景**：
    - Windows API 集成（Windows 原生使用 UTF-16）；
    - 需要处理宽字符的遗留系统。

### 3 sqlite3_create_function_v2()
#### 基本信息

```
int sqlite3_create_function_v2(
  sqlite3 *db,            /* 数据库连接 */
  const char *zFuncName,  /* 函数名 (UTF-8) */
  int nArg,               /* 参数个数 */
  int eTextRep,           /* 首选文本编码 */
  void *pApp,             /* 应用数据指针 */
  void (*xFunc)(sqlite3_context*,int,sqlite3_value**),
  void (*xStep)(sqlite3_context*,int,sqlite3_value**),
  void (*xFinal)(sqlite3_context*),
  void (*xDestroy)(void*)  /* 析构函数 */
);
```

#### 特点
- **增强版本**：在基础版本上增加了析构函数。
- **自动清理**：通过 `xDestroy` 回调自动清理 `pApp` 数据。
- **内存安全**：避免内存泄漏，更加安全。
- **推荐使用**：SQLite 官方推荐使用此版本。

### 4 sqlite3_create_window_function
#### 基本信息
```
int sqlite3_create_window_function(
  sqlite3 *db,            /* 数据库连接 */
  const char *zFuncName,  /* 函数名 */
  int nArg,               /* 参数个数 */
  int eTextRep,           /* 首选文本编码 */
  void *pApp,             /* 应用数据指针 */
  void (*xStep)(sqlite3_context*,int,sqlite3_value**), /* step 函数 */
  void (*xFinal)(sqlite3_context*),                    /* finalize 函数 */
  void (*xValue)(sqlite3_context*),                    /* value 函数 */
  void (*xInverse)(sqlite3_context*,int,sqlite3_value**), /* inverse 函数 */
  void (*xDestroy)(void*)                              /* 析构函数 */
);
```

#### 特点
- **窗口函数专用**：专门用于注册窗口函数（SQLite 3.25.0+）。
- **额外回调**：需实现更多回调函数。
- **反向处理**：支持 `xInverse` 用于优化窗口函数性能。
- **复杂功能**：支持滑动窗口、排序、分组等高级功能。

# 3 UDF 类型
## 3.1 标量函数 (Scalar Functions)

- 接受参数，返回单个值。
- 对每一行数据独立计算。
- 示例：`UPPER()`, `ABS()`, `DATE()`。

## 3.2 聚合函数 (Aggregate Functions)

- 跨多行数据操作，返回聚合结果。
- 需要维护状态信息。
- 示例：`SUM()`, `AVG()`, `COUNT()`。

## 3.3 窗口函数 (Window Functions)

- SQLite 3.25.0+ 支持。
- 在窗口帧上执行计算。
- 示例：`ROW_NUMBER()`, `RANK()`。

# 4 参数含义

```
int sqlite3_create_function_v2(
  sqlite3 *db,            /* 数据库连接 */
  const char *zFuncName,  /* 函数名 (UTF-8) */
  int nArg,               /* 参数个数 */
  int eTextRep,           /* 首选文本编码 */
  void *pApp,             /* 应用数据指针 */
  void (*xFunc)(sqlite3_context*,int,sqlite3_value**),
  void (*xStep)(sqlite3_context*,int,sqlite3_value**),
  void (*xFinal)(sqlite3_context*),
  void (*xDestroy)(void*)  /* 析构函数 */
);
```

## 4.1 zFunctionName

1. 该参数传入函数名：第二个参数是正在创建的 SQL 函数的名称。名称通常为 UTF8 格式，但 `sqlite3_create_function16()` 要求名称采用本地字节序的 UTF16。
2. 函数名长度限制为 255 字节：SQL 函数名的最大长度为 255 字节（UTF8）。若尝试创建更长的名称，将返回 `SQLITE_MISUSE` 错误。
3. UDF 注册支持重载：同一函数名可多次调用该创建接口。例如，若两次调用具有相同函数名但参数个数不同，则会注册该 SQL 函数的两个变体，各接受不同数量的参数。

## 4.2 nArg
1. 表示参数个数，类型为 `int`。
2. 参数个数取值范围为 -1（表示可变），默认为 127（最大值为 `SQLITE_MAX_FUNCTION_ARG`，即 32767）。

## 4.3 eTextRep 文本编码
- **类型**：`int`，位掩码。
- **作用**：指定函数的文本编码偏好。
- **取值**：
    1. `SQLITE_UTF8` (0x01)：UTF-8 编码。
    2. `SQLITE_UTF16LE` (0x02)：UTF-16 小端序。
    3. `SQLITE_UTF16BE` (0x03)：UTF-16 大端序。
    4. `SQLITE_UTF16` (0x04)：使用本地字节序的 UTF-16。
    5. `SQLITE_ANY`：接受任何编码（已废弃）。
    6. **组合标志**：可与确定性标志组合使用。

## 4.4 pApp
### 详细说明
- **类型**：`void*`，任意类型指针。
- **作用**：传递用户自定义数据到函数。
- **生命周期**：由 `xDestroy` 回调管理。
- **使用场景**：
    1. 传递配置参数。
    2. 传递共享资源（如缓存、文件句柄）。
    3. 传递状态信息。

## 4.5 xFunc 标量函数指针
### 详细说明
- **类型**：函数指针 `void (*)(sqlite3_context*, int, sqlite3_value**)`。
- **作用**：标量函数的实现。
- **调用时机**：每次 SQL 查询需要函数值时调用。
- **参数**：
    1. `sqlite3_context*`：函数上下文，用于设置结果和错误。
    2. `int`：实际参数个数。
    3. `sqlite3_value**`：参数值数组。