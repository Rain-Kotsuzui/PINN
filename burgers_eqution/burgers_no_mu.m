function burgers_no_mu(X, T, grid)
    % Solves the inviscid Burgers' equation u_t + u*u_x = 0
    % with corrected boundary handling for the method of characteristics.
    
        % --- 1. 设置默认参数 ---
        if nargin < 1 || isempty(X)
            X = 2;
        end
        if nargin < 2 || isempty(T)
            T = 3;
        end
        if nargin < 3 || isempty(grid)
            grid = 101; % 使用奇数网格点可以确保 x=1 是一个网格点
        end
    
        % --- 2. 创建时空网格 ---
        x = linspace(0, X, grid);
        t = linspace(0, T, grid);
        
        u = zeros(length(t), length(x));
        
        t_shock = 1/pi;
        
        fprintf('开始计算无粘性解析解...\n');
        fprintf('激波形成于 t = %f\n', t_shock);
    
        % --- 3. 逐点计算解析解 ---
        for j = 1:length(t)
            current_t = t(j);
            
            for i = 1:length(x)
                current_x = x(i);
                
                % 处理已知的边界条件
                if current_x == 0 || current_x == X
                    u(j,i) = 0;
                    continue;
                end
    
                g = @(x0) x0 - current_t * sin(pi * (x0 - 1)) - current_x;
                
                try
                    if current_t < t_shock
                        x0_sol = fzero(g, current_x);
                    else
                        if current_x < 1
                            x0_sol = fzero(g, [0, 1]);
                        elseif current_x > 1
                            x0_sol = fzero(g, [1, X]);
                        else % current_x == 1
                            u(j,i) = 0;
                            continue;
                        end
                    end
                    
                    u(j, i) = -sin(pi * (x0_sol - 1));
                    
                catch ME
                    % --- 这是核心修正 ---
                    % 如果 fzero 找不到解，说明该点的特征线已经越过激波
                    % 或者来自边界之外，物理上该点的值应为0。
                    if strcmp(ME.identifier, 'MATLAB:fzero:ValuesAtEndPtsSameSign')
                        u(j, i) = 0; % 将值设置为0
                    else
                        % 对于其他未知错误，仍然发出警告
                        warning('在 t=%f, x=%f 处发生未知错误: %s', current_t, current_x, ME.message);
                        u(j, i) = NaN;
                    end
                end
            end
        end
        
        fprintf('计算完成。\n');
    
        % --- 4. 保存并绘制结果 ---
        save('burgers_no_mu.mat', 'x', 't', 'u');
        
        figure;
        surf(x, t, u);
        shading interp; 
        title('Inviscid Burgers Equation Analytical Solution (Fixed)');
        xlabel('x'); ylabel('t'); zlabel('u');
        view(30, 45);
    
    end