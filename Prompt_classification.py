Below is a **refined classification framework** that builds on your existing Categories **A, B, C, D**—but adds **extra clarity** to handle:

* **Official overrides** (where your documentation explicitly says something is A/B/C/D).
* **Partial vs. full data** (especially addresses or IDs).
* **Hashed/masked vs. unmasked** data (and whether it’s easily reversible).
* **Tie-break situations** where an attribute might fit more than one category.

The goal is to **minimize misclassifications** by being explicit about these nuances.

---

## 1. Final Four Categories

**Category A**: Direct CID — A **single** attribute **by itself** clearly identifies a client.
**Category B**: Indirect CID — An attribute **requires** cross-reference or a lookup to identify a client.
**Category C**: Sensitive by Combination/Context — The attribute **can** identify someone **only** when combined with other data, or is compliance-sensitive (e.g., PEP flag, partial date).
**Category D**: Non-Sensitive — The attribute is **deliberately anonymized/pseudonymized** such that **no** direct or trivial cross-reference can re-identify the client.

---

## 2. Fine-Tuned Classification Steps

Below is a **priority-based** or **top-to-bottom** set of rules that you can apply to any new or unseen attribute. This order helps ensure you pick up official or known “special” classifications first.

### **Step 1. Check for Official Overrides**

1. If your documentation’s “master list” (or Column C/D) explicitly says an attribute is **always** Category A/B/C/D, that override takes **precedence** over everything else.

   * **Example**: “IBAN is always Category B,” “Passport Number is always Category A,” etc.

2. If there’s no official override, continue to Step 2.

### **Step 2. Look for “NSI-…,” “IPID-…,” or Clear “Hashed/Masked” Label**

1. If the attribute or description **explicitly** states it’s **pseudonymized** or “NSI- / IPID-,” or that it’s **fully irreversibly hashed** (i.e. cannot be decoded even by the bank), then classify as **Category D**.
2. **Exception**: If the description says it’s “masked but easily unmasked in normal workflows,” or “encrypted in a standard way that staff can decode,” that might be **Category B** instead, because it’s effectively an **indirect** ID that can be reversed with standard internal tools.

### **Step 3. Check for “Full” Personal or Entity Data** (Potential Category A)

1. If the attribute is any **unmasked, direct** personal or entity detail—e.g., full name, real address (street & building), official ID numbers like passport or driver’s license, unmasked personal email (`[email protected]`)—**Category A**.
2. Rationale: Because a single item alone can identify that client.

### **Step 4. Determine if it’s an ID/Reference Code** (Potential Category B)

1. If the attribute is described as a code, reference, or numeric/alphanumeric ID (e.g., GMI Account ID, internal “Customer #1234,” “CRM code,” “Tax ID” if your policy lumps it under Indirect, etc.), it likely belongs in **Category B**.
2. Because you typically need a **lookup** or database cross-reference to find the client’s actual name.
3. **Exception**: If that ID is partially masked or “un-reversible,” it might be Category D.

### **Step 5. Check for Partial, Time-Bound, or Compliance Flags** (Potential Category C)

If none of the above apply, **but** the attribute is:

* **Partial** (like “city only,” “postcode only,” partial phone, partial name).
* **Time-bound** (Birth Date, Marriage Date, or just the day/month/year alone).
* **Compliance flag / classification** (PEP = True, NGO, SIAP, partial address flags).
* **Fiscal/transaction** amounts or data not inherently revealing identity alone.

Then it goes to **Category C**. Usually, you need other data points (like a name + “City: Geneva”) to actually identify the client, or it’s compliance-sensitive in context (PEP).

### **Step 6. Tie-Break Rule**

1. If an attribute plausibly fits more than one category, choose the **higher** category in terms of sensitivity.

   * E.g., if you’re unsure whether an attribute is B (indirect) or C (only identifying with other data), **lean** to B if it’s used as an ID.
   * E.g., if something is borderline D (non-sensitive) or C (partial?), confirm if it’s truly “anonymous.” If the bank can look it up trivially, that’s not D.

---

## 3. Additional Clarifications & Examples

### **A. Email Addresses**

* If clearly personal/unmasked (`[email protected]`), that’s **Category A**: it might directly identify someone.
* If it’s generic or heavily masked (`[email protected]`), possibly B (if bank staff can decode it) or D if irreversibly hashed.
* If it’s “partial” or “just the domain,” that might be Category C if it only becomes identifying in combination.

### **B. Addresses**

* **Full private address** (street + building + occupant name, or all essential data) → Category A.
* **Partial** (city, state, zip alone) → Category C (identifies only in combination).
* **Tokenized** or “NSI-IPID-Address ID” → Category D (unless easily reversed in day-to-day usage, in which case it’s B).

### **C. Dates**

* **Birth Date** (day/month/year), “Date of Company Formation,” “Death Date,” etc. → typically **Category C**.
* **Just the birth year** or partial day/month → also Category C.
* If it’s fully masked or stored in a non-reversible hashed format (rare), that might push it to D.

### **D. “Tax ID” or “IBAN”**

* If your official doc says “IBAN → B,” or “Tax ID → A,” follow that official rule.
* If no official override, consider “Tax ID” or “SSN” typically Category A if it’s recognized as a direct official identifier in your jurisdiction. Some organizations treat them as B if they say “You need a lookup.”

### **E. PEP Flag, NGO, SCAP**

* Usually **Category C**: it’s not direct, but it’s highly sensitive once combined with the client identity.

### **F. Non-Sensitive (NSI) Codes**

* If you see “NSI-Non-IPID-…” or “irreversible hash” or “pseudonymized ID with no direct re-mapping,” that’s **Category D**.
* If there is a mention of a normal or widely accessible “decryption key,” it’s actually B.

---

## 4. Putting It All Together

1. **Start** with an official or known “master reference” (e.g., if your doc says “Passport # → A,” no further logic needed).
2. Then look for “NSI-…” or “IPID-…” or “Hashed/Masked.” If truly un-reversible, it’s D. If reversible, likely B.
3. If it’s obviously a real, **unmasked** personal or legal ID, it’s A.
4. Otherwise, if it’s a code or numeric reference, it’s B.
5. If partial data, time-bound, or compliance flags that only matter with additional context, it’s C.
6. Use tie-breaks if needed.

This refined approach addresses the **common pitfalls** (partial vs. full, masked vs. unmasked, official overrides) and should yield more **accurate** classification results aligned with your **Column C & D** expectations.



##################
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
