const msg = document.querySelector("#predict-msg");
// console.log(msg);
percentage = parseFloat(msg.innerHTML)

console.log(percentage);
if (percentage > 50)
    msg.style.backgroundColor = 'red';
else
    msg.style.backgroundColor = 'green';