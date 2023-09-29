import React from "react";

const LoanResult = ({ isApproved }) => {
  return (
    <div>
      {isApproved ? (
        <p>Parabéns, você foi aprovado!</p>
      ) : (
        <p>Infelizmente, você não foi aprovado.</p>
      )}
    </div>
  );
};

export default LoanResult;
