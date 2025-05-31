Below is a **comprehensive “master” classification system** that unifies **Categories A, B, C, and D** from all the specifications you shared. It includes:

1. **Definitions of Categories (A, B, C, D)**
2. **Subcategories** and how they relate to the original detail (like “Natural Person \[i],” “Legal Person \[ii],” “time-bound,” “compliance flags,” etc.)
3. **Step-by-step classification logic** so you can handle **new or unseen attributes**.
4. **Concrete examples** at each step to illustrate how to classify.

Where relevant, I’ve merged and aligned the various versions of Category C (time-bound attributes, combination attributes, compliance flags, etc.) and Category D (non-sensitive identifiers). This final set of rules should be comprehensive enough to classify **any** new data point in a consistent manner.

---

## 1. Top-Level Overview

### **Category A: Direct CID**

> “**Direct** Client Identifier”
> Attributes that, **by themselves**, **immediately identify** a client (whether a **Natural Person** or a **Legal Person**). If you look at that attribute alone, you can figure out **who** the client is.

**Subcategories** (following your original references):

* **\[i] Natural Person**
  E.g. Child Name, Full Name, Signature.
* **\[ii] Legal Person**
  E.g. Company UID, Registered Name, Trading Name, GMI Account ID, Stock Symbol.
* **\[iii] External/Public Registry**
  E.g. Passport Number, Tax ID, Military ID, Legal Entity Identifier (LEI), etc.
* **\[iv] Physical & Electronic Address**
  E.g. Private Address, Business Address, Telephone Number, Email Address, IP Address.
* **\[v] UBS Generated IDs (direct)**
  E.g. ETD Financing Account **Name** (when it directly includes the client name).

In short: **If a single data point alone reveals someone’s identity**, it is **Category A**.

---

### **Category B: Indirect Sensitive IDs for CID**

> “**Indirect** Client Identifier”
> Attributes that can reveal a client’s identity **only if** you have additional knowledge or cross-references. They do not **immediately** disclose who the client is.

**Subcategories** (matching your references):

* **\[iii] Externally Generated IDs** (Indirect)
  E.g. IBAN, EH Client ID, certain external codes that **don’t** show the client’s name but are used to identify them in external systems.
* **\[iv] Physical & Electronic Address (Indirect)**
  E.g. “Email address of a contact person” of a corporate client. You can’t confirm the corporate client’s identity from that alone unless you know the relationship.
* **\[v] UBS Generated IDs (Indirect)**
  E.g. Account Number, Customer Number, Portfolio ID, CRM Code, GMI Account references, “CS ID,” “CIS Code,” etc.

Typical signs of Category B:

1. It’s **some kind of ID or code** (internal or external) that does not embed the client’s name.
2. By itself, it’s **not** obviously a person’s or entity’s identity (like “12345XYZ”), but if you have the internal mapping or external lookup, you can identify the client.

---

### \*\*Category C: Sensitive Client Identifiers from **Combination** or **Context**

> Data that **can** become identifying **when combined** with other data (or used in certain compliance contexts).

In some of your specifications, Category C is further split into:

* **Combination of multiple attributes**

  * E.g. Civil Status, Religious Affiliation, Web Cookie, Risk Tolerance, or partial address info. Each alone isn’t enough to pinpoint someone, but combined with a name or a birth date, it can be.

* **Time-bound attributes**

  * E.g. Birth Date, Birth Year, Marriage Day/Month, Company Liquidation Date. A date alone doesn’t reveal who you are, but if we also know your first name and your birth date, that can be identifying.

* **Fiscal / Transaction attributes**

  * E.g. Gross Amount, Net Amount, Account Currency, Salary, Transaction Type, Value Date. On their own, these do not identify the client, but if you see “John Smith placed a trade on \[X date/time, for Y volume],” it becomes identifying.

* **Compliance/Code/Flag attributes**

  * E.g. “Politically Exposed Person” (PEP) flag, “NGO-NGO,” “NTBR,” “SCAP,” “SIAP,” partial address flags. These flags and partial addresses do not directly reveal the client’s name. But in the context of “**which** client is flagged as PEP?” it can identify them.

**Additionally**, some references mention **CID-\[vi]** for certain compliance flags and partial addresses. That is a finer detail indicating that these data points become “sensitive identifiers” once correlated with a client’s name or ID.

---

### **Category D: Non-Sensitive Identifiers (NSI)**

> **Pseudonymized** or **anonymized** references intentionally used to **avoid** identifying a client.

All the “NSI-…” or “NSI-IPID-…” items from your texts are examples. They **cannot** readily identify someone, even if you have them. Usually, they are random codes with no direct link to a name.

Examples:

* **NSI-IPID-Address ID**: A placeholder for an address.
* **NSI-Non-IPID-AccountID**: A pseudo-account ID that’s not recognized in the real environment as the actual number.
* **NSI-Non-IPID-Client ID**: E.g. a 6-digit random code used in a system that does not map to any known real account or name.
* **NSI-Non-IPID-UBSGUID**: An alpha-numeric reference that is “safe” from a privacy standpoint because no external entity can decode it.

**Key characteristic**: You cannot identify the client from these NSI values **even if** you see them, unless you have extremely special or internal “secret” mapping that typically does not exist outside specialized compliance teams.

Hence, **Category D** = safe placeholders that do not themselves pose a privacy risk and are not considered “sensitive” in normal contexts.

---

## 2. Step-by-Step Classification Logic

To classify **any** new or “unseen” attribute, follow these steps:

1. **Check if the attribute, by itself, identifies the client**

   * If **Yes**, it’s **Category A** (Direct CID).
   * **Example**: A user’s Full Legal Name, a Passport Number on its own.

2. **If No, see if it’s a code or ID used to identify the client**

   * If **it’s a code** that, when cross-referenced with a certain system or database, reveals a client → **Category B** (Indirect).
   * **Example**: Internal “Account Number 1000523” or “CRM Code ABC123.” Alone, it’s not obvious who this is, but with the bank’s internal systems, you can find out.

3. **If No, check if the attribute becomes identifying only in combination with other details** (or if it’s a compliance flag, partial address, time-bound info, etc.)

   * If so, it’s **Category C**.
   * **Example**:

     * “Marital Status: Married.” Not enough alone. But if you also know the person’s name or date of birth, you might identify them.
     * “Politically Exposed Person (PEP) Flag.” On its own, it’s just a label—but combined with “someone in the system is flagged PEP,” it can lead to identification.

4. **If none of the above** (i.e., the attribute is intentionally anonymized or pseudonymized, so it **cannot** be used to identify the client) → **Category D** (Non-Sensitive).

   * **Example**: “NSI-Non-IPID-ClientPointGUID: 79c3a010-8bd1-…” is a random ID with no direct correlation to a known client.

---

## 3. Examples & Explanations

### **Example A: “Full Name: Jane Elizabeth Doe”**

* **Classification**: **Category A** (\[i] Natural Person).
* **Reason**: By itself, it identifies a real person.

### **Example B: “Customer Number = 12345678”**

* **Classification**: **Category B** (\[v] UBS Generated IDs).
* **Reason**: Not obviously “Jane Doe,” but an internal system could map that number to her.

### **Example C: “City of Residence: Geneva”**

* **Classification**: **Category C** (Location/partial address).
* **Reason**: On its own, “Geneva” is not enough to identify the client. But combined with more data (like “Jane Doe in Geneva”), it can become identifying.

### **Example D: “PEP Flag = True”**

* **Classification**: **Category C** (Compliance flags).
* **Reason**: Doesn’t directly say who the person is; but it’s sensitive in combination with a name or client record.

### **Example E: “NSI-Non-IPID-Client ID = BCCH-52739”**

* **Classification**: **Category D** (Non-sensitive pseudonym).
* **Reason**: That ID is specifically described as **non-sensitive** or anonymized. It cannot identify the person by itself.

---

## 4. Final Recap

1. **Category A (Direct)**: Single attribute **by itself** reveals identity.
2. **Category B (Indirect)**: It’s a code or reference that needs cross-reference to identify.
3. **Category C** (Sensitive by **combination**, compliance, partial info, or time-based): Doesn’t identify alone but becomes identifying (or is compliance-sensitive) when combined with other data.
4. **Category D (Non-Sensitive)**: Pseudonymized / anonymized references that are **not** directly or indirectly revealing the client’s identity, typically used as “safe” placeholders.

This integrated structure **captures every detail** from your prior specifications. You can apply it to any **new** or **unseen** data simply by asking: “Does it directly reveal identity? Indirectly? Only in combination or for compliance? Or is it a deliberately non-sensitive stand-in?” and then assign **A**, **B**, **C**, or **D** accordingly.
