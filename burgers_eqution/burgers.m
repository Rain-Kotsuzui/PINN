
function burgers(NU,X,T,grid)
    if nargin < 1 || isempty(NU)
        NU = 0.01;
    end
    if nargin < 2 || isempty(X)
        X = 2;
    end
    if nargin < 3 || isempty(T)
        T = 3;
    end
    if nargin < 4 || isempty(grid)
        grid = 100;
    end


    nu = NU; 
    m = 0;
    x = linspace(0, X, grid);
    t = linspace(0, T, grid);

    sol = pdepe(m, @burgers_pde, @burgers_ic, @burgers_bc, x, t, [], nu);

    u = sol(:, :, 1);
    save('burgers_matlab.mat', 'x', 't', 'u');
    figure;
    surf(x, t, u);
    shading interp; 
    title(['Burgers Equation Solution (nu = ' num2str(nu) ')']);
    xlabel('x'); ylabel('t'); zlabel('u');

    % figure(2);
    % hold on;

    % for i = 1:length(t)
    %     xlim([0 X]);
    %     ylim([-1.5 1.5]);
    %     cla;
    %     plot(x, u(i, :), 'DisplayName', ['t = ' num2str(t(i))]);
    %     pause(0.01);
    % end

end


function [c, f, s] = burgers_pde(~, ~, u, DuDx, nu)
    c = 1;
    f = nu * DuDx - 0.5 * u ^ 2;
    s = 0;
end

function u0 = burgers_ic(x, ~)
    u0 = -sin(pi * (x - 1));
end

function [pl, ql, pr, qr] = burgers_bc(~, ul, ~, ur, ~, ~)
    pl = ul; % p(xl,t,ul)
    ql = 0; % q(xl,t)
    pr = ur; % p(xr,t,ur)
    qr = 0; % q(xr,t)
end
