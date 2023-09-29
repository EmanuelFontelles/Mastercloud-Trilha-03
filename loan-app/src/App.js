import React, { useState } from "react";
import LoanForm from "./components/LoanForm";
import LoanResult from "./components/LoanResult";

function App() {
  const [result, setResult] = useState(null);

  const handleResult = (isApproved) => {
    setResult(isApproved);
  };

  return (
    <div>
      <h1>Empréstimo Pessoal</h1>
      <LoanForm onResult={handleResult} />
      {result !== null && <LoanResult isApproved={result} />}
    </div>
  );
}

export default App;
