import React, { useState } from "react";
import axios from "axios";

const LoanForm = ({ onResult }) => {
  const [income, setIncome] = useState("");

  const checkApproval = async () => {
    try {
      const response = await axios.post("http://your-api.com/check", { income });
      onResult(response.data.isApproved);
    } catch (error) {
      console.error("API call failed", error);
      onResult(false);
    }
  };

  return (
    <div>
      <label>Informe sua renda: </label>
      <input
        type="text"
        value={income}
        onChange={(e) => setIncome(e.target.value)}
      />
      <button onClick={checkApproval}>Verificar Aprovação</button>
    </div>
  );
};

export default LoanForm;
